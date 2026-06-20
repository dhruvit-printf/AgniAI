"""
schemas.py
==========
Pydantic v2 models for strict request/response validation across all stages of the AgniAI pipeline.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class IntentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    category: Optional[str] = None
    subcategory: Optional[str] = None
    confidence: Any = None
    operation: Optional[str] = None
    number: Optional[int] = None
    section: Optional[str] = None
    sub_section: Optional[str] = None
    grading: Optional[str] = None
    leave_type: Optional[str] = None
    sport: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    unit_name: Optional[str] = None
    attempt_no: Optional[int] = None
    from_attempt: Optional[int] = None
    to_attempt: Optional[int] = None
    date: Optional[str] = None
    item_name: Optional[str] = None
    item_category: Optional[str] = None
    company_id: Optional[int] = None
    platoon_id: Optional[int] = None
    batch_id: Optional[int] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    agniveer_no: Optional[str] = None
    bmi_category: Optional[str] = None
    medical_status: Optional[str] = None
    widget_hint: Optional[str] = None
    widgetHint: Optional[str] = None

class DotNetPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    category: Optional[str] = None
    operation: Optional[str] = None
    n: Optional[int] = None
    section: Optional[str] = None
    subSection: Optional[str] = None
    grading: Optional[str] = None
    leaveType: Optional[str] = None
    sport: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    unitName: Optional[str] = None
    attemptNo: Optional[int] = None
    fromAttempt: Optional[int] = None
    toAttempt: Optional[int] = None
    date: Optional[str] = None
    itemName: Optional[str] = None
    itemCategory: Optional[str] = None
    companyId: Optional[int] = None
    platoonId: Optional[int] = None
    batchId: Optional[int] = None
    fromDate: Optional[str] = None
    toDate: Optional[str] = None
    agniveerNo: Optional[str] = None
    bmiCategory: Optional[str] = None
    medicalStatus: Optional[str] = None
    fullName: Optional[str] = None
    groupBy: Optional[str] = None
    analyticsHint: Optional[str] = None

class DotNetResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: Optional[List[Dict[str, Any]]] = None
    data: Optional[Any] = None
    status: Optional[bool] = None
    message: Optional[str] = None
    raw_response: Optional[Any] = None

class CombinedResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: List[Dict[str, Any]] = []
    sides: Optional[List[Dict[str, Any]]] = None
    comparison: Optional[Dict[str, Any]] = None
    sections: Optional[List[Dict[str, Any]]] = None
    chartData: Optional[List[Dict[str, Any]]] = None
    granularity: Optional[str] = None
    trendDirection: Optional[str] = None
    labels: Optional[List[str]] = None
    values: Optional[List[Any]] = None
    groupBy: Optional[str] = None
    degraded: Optional[bool] = None
    failedFilters: Optional[List[str]] = None
    matchCount: Optional[int] = None
    totalBeforeFilter: Optional[int] = None

class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: List[str] = []
    insights: List[str] = []
    summary: str = ""

class PredictionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend: str = ""
    forecast: str = ""
    shortTerm: Optional[str] = None
    futureTrends: Optional[List[str]] = None

class ConclusionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    message: Optional[str] = None

class SuggestedQuestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str

class WidgetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: str = ""
    widgetType: str = ""
    type: Optional[str] = None  # For backward compatibility

class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str = ""
    traceId: str = ""
    sessionId: str = ""
    executionTimeMs: int = 0
    intentDurationMs: int = 0
    dotnetDurationMs: int = 0
    combineDurationMs: int = 0
    analysisDurationMs: int = 0
    predictionDurationMs: int = 0
    conclusionDurationMs: int = 0
    totalDurationMs: int = 0
    # Backward compatibility fields
    confidence: Optional[float] = None
    queryType: Optional[str] = None
    operationCount: Optional[int] = None
    planner_duration: Optional[float] = None
    intent_duration: Optional[float] = None
    dotnet_duration: Optional[float] = None
    combiner_duration: Optional[float] = None
    report_duration: Optional[float] = None
    total_duration: Optional[float] = None

class FinalResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: bool
    queryType: Optional[str] = None
    introMessage: Dict[str, Any] = {}
    formattedData: Dict[str, Any] = {}
    answer: Optional[Dict[str, Any]] = None
    analysis: Optional[AnalysisModel] = None
    prediction: Optional[PredictionModel] = None
    conclusion: Optional[ConclusionModel] = None
    suggestedQuestions: List[str] = []
    widgets: List[WidgetModel] = []
    metadata: MetadataModel
    overallConfidence: float = 0.0
    partialFailure: bool = False
    failedSections: List[str] = []
    
    # Backward compatibility fields
    intent: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    sessionId: Optional[str] = None
