from fastapi import APIRouter, HTTPException, Request

from api.schemas import ABStatsResponse, DriftResponse
from src.monitoring.drift import (
    get_confidence_drift,
    get_oos_rate_drift,
    get_drift_summary,
    load_current_data,
    load_reference_data,
    run_drift_report,
)

router = APIRouter()


@router.get("/monitoring/drift", response_model=DriftResponse)
def get_drift() -> DriftResponse:
    try:
        reference_df = load_reference_data()
        current_df = load_current_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    snapshot = run_drift_report(reference_df, current_df)
    drift_summary = get_drift_summary(snapshot)
    confidence_drift = get_confidence_drift(reference_df, current_df)
    oos_rate_drift = get_oos_rate_drift(reference_df, current_df)

    return DriftResponse(
        drift_summary=drift_summary,
        confidence_drift=confidence_drift,
        oos_rate_drift=oos_rate_drift,
    )


@router.get("/monitoring/ab-stats", response_model=ABStatsResponse)
def get_ab_stats(request: Request) -> ABStatsResponse:
    ab_router = request.app.state.ab_router
    stats = ab_router.get_stats()
    return ABStatsResponse(**stats)


@router.post("/monitoring/reset")
def reset_monitoring(request: Request) -> dict:
    ab_router = request.app.state.ab_router
    ab_router.reset_stats()
    return {"status": "reset"}