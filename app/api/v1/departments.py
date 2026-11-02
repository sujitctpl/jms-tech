import uuid
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.department import DepartmentCreate, DepartmentResponse, DepartmentUpdate
from app.repositories.department_repo import DepartmentRepository

router = APIRouter(prefix="/api/v1/departments", tags=["Departments & Zones"])


@router.get("", response_model=List[DepartmentResponse], summary="List all detection departments")
def list_departments(
    camera_id: Optional[uuid.UUID] = Query(None, description="Filter by camera UUID"),
    only_active: bool = Query(False, description="Filter only active departments"),
    db: Session = Depends(get_db),
) -> Any:
    repo = DepartmentRepository(db)
    repo.seed_defaults_if_empty()
    return repo.list_departments(camera_id=camera_id, only_active=only_active)


@router.post("", response_model=DepartmentResponse, status_code=status.HTTP_201_CREATED, summary="Create new detection department")
def create_department(
    data: DepartmentCreate,
    db: Session = Depends(get_db),
) -> Any:
    repo = DepartmentRepository(db)
    return repo.create_department(data)


@router.patch("/{department_id}", response_model=DepartmentResponse, summary="Update department settings")
def update_department(
    department_id: str,
    data: DepartmentUpdate,
    db: Session = Depends(get_db),
) -> Any:
    repo = DepartmentRepository(db)
    dept = repo.update_department(department_id, data)
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    return dept


@router.delete("/{department_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete department")
def delete_department(
    department_id: str,
    db: Session = Depends(get_db),
) -> None:
    repo = DepartmentRepository(db)
    ok = repo.delete_department(department_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Department not found")

