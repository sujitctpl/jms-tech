from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.visitor import (
    VisitorDetectRequest,
    VisitorDetectResponse,
    VisitorListResponse,
    VisitorResponse,
    VisitorTagRequest,
    VisitorZoneUpdateRequest,
    WalkInCreateRequest,
)
from app.services.visitor_service import VisitorService

router = APIRouter(prefix="/api/v1/visitors", tags=["Visitors & Visit Purpose"])


@router.get(
    "",
    response_model=VisitorListResponse,
    status_code=status.HTTP_200_OK,
    summary="List visitors for Visit Purpose Management",
)
def list_visitors(
    tagging_status: Optional[str] = Query(None, description="'Tagged', 'Untagged', or 'all'"),
    visit_purpose: Optional[str] = Query(None, description="Optional filter by purpose"),
    db: Session = Depends(get_db),
) -> VisitorListResponse:
    """Retrieve all detected visitors with their photos, timestamps, and tagging statuses."""
    service = VisitorService(db)
    return service.list_visitors(tagging_status=tagging_status, visit_purpose=visit_purpose)


@router.post(
    "/detect",
    response_model=VisitorDetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest face detection event from camera/edge detector",
)
def process_face_detection(
    payload: VisitorDetectRequest,
    db: Session = Depends(get_db),
) -> VisitorDetectResponse:
    """
    Called when a face is detected:
    - Filters out showroom employees (records attendance instead).
    - Re-identifies returning visitors across cameras (maintains same ID).
    - Creates new visitor with face crop thumbnail for brand new walk-ins.
    """
    service = VisitorService(db)
    return service.process_detection(payload)


@router.patch(
    "/{person_id}/tag",
    response_model=VisitorResponse,
    status_code=status.HTTP_200_OK,
    summary="Tag visitor with visit purpose, salesperson, and notes",
)
def tag_visitor(
    person_id: str,
    payload: VisitorTagRequest,
    db: Session = Depends(get_db),
) -> VisitorResponse:
    """Assign visit purpose intent and staff to a customer record."""
    service = VisitorService(db)
    return service.tag_visitor(person_id=person_id, payload=payload)


@router.post(
    "/update-zone",
    response_model=Optional[VisitorResponse],
    status_code=status.HTTP_200_OK,
    summary="Update live visitor zone dwell times and departure",
)
def update_visitor_zone(
    payload: VisitorZoneUpdateRequest,
    db: Session = Depends(get_db),
) -> Optional[VisitorResponse]:
    """Updates active zone dwell times, departure time, and primary zone for a visitor."""
    service = VisitorService(db)
    res = service.update_visitor_zone(payload)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visitor record not found for person_id '{payload.person_id}' or tracker '{payload.tracker_id}'",
        )
    return res


@router.post(
    "/walk-in",
    response_model=VisitorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Manually log walk-in visitor from reception",
)
def create_manual_walk_in(
    payload: WalkInCreateRequest,
    db: Session = Depends(get_db),
) -> VisitorResponse:
    """Manually add a visitor to the Visit Purpose table."""
    service = VisitorService(db)
    return service.create_walk_in(payload)


@router.post(
    "/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear all visitors and reset visitor records",
)
def clear_all_visitors(
    db: Session = Depends(get_db),
):
    """Deletes all visitor records and cleans up stored visitor snapshot images."""
    service = VisitorService(db)
    return service.clear_all_visitors()


@router.delete(
    "/{person_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single bounced or cancelled visitor record",
)
def delete_single_visitor(
    person_id: str,
    db: Session = Depends(get_db),
):
    """Deletes a single visitor record, footfall event, and snapshot image."""
    service = VisitorService(db)
    success = service.delete_visitor(person_id)
    return {"success": success, "person_id": person_id}


@router.delete(
    "",
    status_code=status.HTTP_200_OK,
    summary="Delete all visitor records",
)
def delete_all_visitors(
    db: Session = Depends(get_db),
):
    """Deletes all visitor records and cleans up stored visitor snapshot images."""
    service = VisitorService(db)
    return service.clear_all_visitors()


