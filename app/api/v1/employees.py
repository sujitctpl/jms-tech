from datetime import date
from typing import Any, List, Optional
import uuid
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.employee import (
    AttendancePunchRequest,
    AttendancePunchResponse,
    DailyAttendanceReportResponse,
    EmployeeCreate,
    EmployeeResponse,
    EmployeeUpdate,
    FaceEnrollmentRequest,
)
from app.services.employee_service import EmployeeService

router = APIRouter(tags=["Employee Management & Attendance"])


@router.post(
    "/api/v1/employees",
    response_model=EmployeeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new showroom employee",
)
def register_employee(
    employee_in: EmployeeCreate,
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    return service.register_employee(employee_in)


@router.get(
    "/api/v1/employees",
    response_model=List[EmployeeResponse],
    summary="List registered employees",
)
def list_employees(
    department: Optional[str] = Query(None, description="Filter by department"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    return service.list_employees(department=department, is_active=is_active, skip=skip, limit=limit)


@router.get(
    "/api/v1/employees/{employee_id}",
    response_model=EmployeeResponse,
    summary="Get employee profile",
)
def get_employee(
    employee_id: str,
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    return service.get_employee(employee_id)


@router.patch(
    "/api/v1/employees/{employee_id}",
    response_model=EmployeeResponse,
    summary="Update employee profile or active status",
)
def update_employee(
    employee_id: str,
    employee_in: EmployeeUpdate,
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    return service.update_employee(employee_id, employee_in)


@router.post(
    "/api/v1/employees/{employee_id}/face",
    response_model=EmployeeResponse,
    summary="Enroll or update employee biometric face embedding via Base64 or vector",
)
def enroll_employee_face(
    employee_id: str,
    req: FaceEnrollmentRequest = Body(...),
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    emb = req.embedding_vector if req.embedding_vector is not None else req.embedding
    return service.enroll_face(
        employee_id=employee_id,
        face_image_base64=req.face_image_base64,
        face_images_base64=req.face_images_base64,
        embedding_vector=emb,
    )


@router.post(
    "/api/v1/employees/{employee_id}/face/upload",
    response_model=EmployeeResponse,
    summary="Enroll employee face by uploading one or more photo files (JPG/PNG). The 1st photo is kept as profile picture.",
)
async def upload_employee_face_image(
    employee_id: str,
    files: Optional[List[UploadFile]] = File(None, description="One or more employee portrait images (1st kept as profile picture)"),
    file: Optional[UploadFile] = File(None, description="Single employee portrait image (legacy support)"),
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    upload_list: List[UploadFile] = []
    if files:
        upload_list.extend([f for f in files if f and f.filename])
    if file and file.filename:
        upload_list.append(file)

    if not upload_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one valid image file must be uploaded.",
        )

    contents_list = [await f.read() for f in upload_list]
    return service.enroll_faces_from_bytes(
        employee_id=employee_id,
        images_bytes=contents_list,
    )


@router.post(
    "/api/v1/attendance/punch",
    response_model=AttendancePunchResponse,
    status_code=status.HTTP_200_OK,
    summary="Record employee check-in or check-out",
)
def record_attendance_punch(
    punch: AttendancePunchRequest,
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    return service.record_punch(punch)


@router.get(
    "/api/v1/attendance/daily",
    response_model=DailyAttendanceReportResponse,
    summary="Get daily showroom employee attendance report",
)
def get_daily_attendance_report(
    attendance_date: Optional[date] = Query(None, description="Report date (defaults to today)"),
    department: Optional[str] = Query(None, description="Filter by department"),
    auto_mark_absent: bool = Query(True, description="Automatically mark absent for employees with no punch"),
    db: Session = Depends(get_db),
) -> Any:
    service = EmployeeService(db)
    target_date = attendance_date or date.today()
    return service.get_daily_report(
        report_date=target_date,
        department=department,
        auto_mark_absent=auto_mark_absent,
    )
