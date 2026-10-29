"""Enroll an employee's face embedding from one or more image files (JPG / PNG).
Can register a new employee or update an existing employee's biometric face.
The 1st photo is kept as their profile picture.

Usage:
    python scripts/enroll_employee_image.py --id EMP-101 --image "path/to/portrait.jpg"
    python scripts/enroll_employee_image.py --id EMP-101 --image "path/to/front.jpg" "path/to/angle.jpg"
    python scripts/enroll_employee_image.py --id EMP-101 --name "Amit Roy" --dept "Sales" --image "photo1.jpg" "photo2.jpg"
"""

import argparse
import os
import shutil
import sys
from typing import List, Union
import cv2
import numpy as np

# Ensure app is discoverable on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import SessionLocal
from app.repositories.employee_repo import EmployeeRepository
from app.vision.face_embedder import FaceEmbedder
from app.vision.vector_store import PgVectorStoreManager


def enroll_employee_photos(
    employee_id: str,
    image_paths: Union[str, List[str]],
    name: str = None,
    department: str = "General",
    designation: str = "Staff",
):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if not image_paths:
        print("[-] Error: No image paths provided.")
        return False

    valid_paths = []
    for p in image_paths:
        if os.path.exists(p):
            valid_paths.append(p)
        else:
            print(f"[-] Warning: Image file not found: {p}")

    if not valid_paths:
        print("[-] Error: None of the provided image paths exist.")
        return False

    print(f"[*] Processing {len(valid_paths)} images for employee {employee_id}...")
    print(f"[*] 1st image will be kept as profile picture: {valid_paths[0]}")

    embedder = FaceEmbedder(min_size=36, min_confidence=0.5)
    embeddings = []

    for idx, img_path in enumerate(valid_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[-] Warning: OpenCV failed to read image #{idx + 1}: {img_path}")
            continue
        try:
            emb = embedder.extract_face_embedding_from_image(img)
            embeddings.append(emb)
            print(f"[+] Extracted 512D ArcFace vector from image #{idx + 1}: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"[-] Warning: Face detection/embedding failed on image #{idx + 1}: {e}")

    if not embeddings:
        print("[-] Error: Could not extract face embeddings from any of the provided images.")
        return False

    # Combine embeddings (centroid vector)
    if len(embeddings) == 1:
        final_embedding = embeddings[0]
    else:
        mean_vec = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_vec)
        final_embedding = (mean_vec / norm).astype(np.float32) if norm > 0 else mean_vec.astype(np.float32)
        print(f"[+] Combined {len(embeddings)} face vectors into an aggregated ArcFace profile template.")

    # Save 1st photo as profile picture
    first_path = valid_paths[0]
    upload_dir = os.path.join("static", "uploads", "employees")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(first_path)[1].lower() or ".jpg"
    clean_id = str(employee_id).replace("/", "_").replace("\\", "_")
    dest_filename = f"{clean_id}_profile{ext}"
    dest_path = os.path.join(upload_dir, dest_filename)
    shutil.copyfile(first_path, dest_path)
    photo_url = f"/static/uploads/employees/{dest_filename}"
    print(f"[+] Saved 1st photo as profile picture: {photo_url}")

    db = SessionLocal()
    try:
        emp_repo = EmployeeRepository(db)
        emp = emp_repo.get_by_employee_id(employee_id)

        if not emp:
            emp_name = name or employee_id
            print(f"[*] Employee '{employee_id}' not found in database. Creating new record for '{emp_name}'...")
            emp = emp_repo.create(
                employee_id=employee_id,
                name=emp_name,
                department=department,
                designation=designation,
                embedding=final_embedding,
                photo_url=photo_url,
            )
            print(f"[+] Created employee: {emp.name} (ID: {emp.employee_id})")
        else:
            emp_repo.update_face_embedding(emp, final_embedding, photo_url=photo_url)
            print(f"[+] Updated face embedding and profile picture for employee: {emp.name} (ID: {emp.employee_id})")

        # Verify searchability immediately via pgvector
        try:
            vec_mgr = PgVectorStoreManager(db=db)
            matched_id, score = vec_mgr.find_best_employee_match(final_embedding, similarity_threshold=0.50, db=db)
            print(f"[+] pgvector verification: Matched {matched_id} with cosine similarity = {score:.4f}")
        except Exception as e:
            print(f"[*] Note on pgvector verification: {e}")

        print(f"\n[SUCCESS] Employee face enrolled with {len(embeddings)} sample(s), profile picture set!")
        return True

    finally:
        db.close()


# Backward-compatible alias
enroll_employee_photo = enroll_employee_photos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll employee face from image file(s) into pgvector. 1st image is kept as profile picture.")
    parser.add_argument("--id", required=True, help="Employee ID (e.g. EMP-101)")
    parser.add_argument("--image", nargs="+", required=True, help="Path to one or more portrait image files (JPG or PNG). 1st is profile picture.")
    parser.add_argument("--name", default=None, help="Employee Full Name (if creating a new record)")
    parser.add_argument("--dept", default="General", help="Department (e.g. Sales, Jewelry, Accounts)")
    parser.add_argument("--designation", default="Staff", help="Designation (e.g. Sales Executive)")

    args = parser.parse_args()
    enroll_employee_photos(
        employee_id=args.id,
        image_paths=args.image,
        name=args.name,
        department=args.dept,
        designation=args.designation,
    )

