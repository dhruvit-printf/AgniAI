"""
schemas.py
==========
Pydantic v2 models for strict request/response validation across all stages of the AgniAI pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class IntentModel(_BaseModel):
    # extra="ignore": understand_query adds "query_type" to the intent dict;
    # any other unexpected keys from future classify_admin_intent changes are
    # silently dropped rather than causing drift warnings on every request.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    category: Optional[str] = None
    subcategory: Optional[str] = None
    confidence: Optional[Union[str, float]] = None
    operation: Optional[str] = None
    raw_query: Optional[str] = None
    query_type: Optional[str] = None  # ← added: set by understand_query()
    number: Optional[int] = None
    section: Optional[str] = None
    sub_section: Optional[str] = None
    metric: Optional[str] = None
    comparison_target: Optional[str] = None
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
    date_range: Optional[Any] = None
    agniveer_no: Optional[str] = None
    bmi_category: Optional[str] = None
    blood_group: Optional[str] = None
    type: Optional[str] = None
    medical_status: Optional[str] = None
    diagnose: Optional[str] = None
    group_by: Optional[str] = None
    sort_by: Optional[str] = None
    aggregation: Optional[str] = None
    top_n: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None
    widget_hint: Optional[str] = None
    widgetHint: Optional[str] = None
    days: Optional[int] = None


class DotNetPayloadModel(_BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    commandId: Optional[int] = 0
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
    bloodGroup: Optional[str] = None
    equipmentName: Optional[str] = None
    equipmentType: Optional[str] = None
    companyId: Optional[int] = None
    platoonId: Optional[int] = None
    batchId: Optional[int] = None
    fromDate: Optional[str] = None
    toDate: Optional[str] = None
    agniveerNo: Optional[str] = None
    bmiCategory: Optional[str] = None
    medicalStatus: Optional[str] = None
    diagnose: Optional[str] = None
    days: Optional[int] = None
    fullName: Optional[str] = None
    groupBy: Optional[str] = None
    analyticsHint: Optional[str] = None
    sortBy: Optional[str] = None  # ← added: passed by admin_pipeline for ranking
    returnCondition: Optional[str] = None
    givenCondition: Optional[str] = None


class DotNetResponseModel(_BaseModel):

    success: Optional[bool] = None
    commandLabel: Optional[str] = None
    status: Optional[bool] = None
    data: Optional[Any] = None
    message: Optional[str] = None
    records: Optional[List[Dict[str, Any]]] = None


class CombinedResponseModel(_BaseModel):

    success: Optional[bool] = None
    status: Optional[bool] = None
    commandLabel: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Any] = None
    queryType: Optional[str] = None
    sectionCount: Optional[int] = None
    increase: Optional[bool] = None
    decrease: Optional[bool] = None
    stable: Optional[bool] = None
    left: Optional[Dict[str, Any]] = None
    right: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    comparisonMetrics: Optional[Dict[str, Any]] = None
    comparedMetrics: Optional[List[str]] = None
    records: List[Dict[str, Any]] = Field(default_factory=list)
    sides: Optional[List[Dict[str, Any]]] = None
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
    filterDepth: Optional[int] = None


class AnalysisModel(_BaseModel):

    insights: List[str] = Field(default_factory=list)
    summary: str = ""
    statistics: Dict[str, Any] = Field(default_factory=dict)
    observations: List[str] = Field(default_factory=list)
    predictions: List[str] = Field(default_factory=list)


class PredictionModel(_BaseModel):

    trend: str = ""
    projection: Optional[str] = None
    heuristicEstimate: Optional[str] = None
    shortTerm: Optional[str] = None
    futureTrends: List[str] = Field(default_factory=list)


class ConclusionModel(_BaseModel):

    summary: str = ""
    bullets: List[str] = Field(default_factory=list)


class SuggestedQuestionModel(_BaseModel):

    question: str


class WidgetModel(_BaseModel):

    section: str = ""
    widgetType: str = ""
    type: Optional[str] = None


class MetadataModel(_BaseModel):

    requestId: str = ""
    traceId: str = ""
    sessionId: str = ""
    timings: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    executionTimeMs: int = 0
    intentDurationMs: int = 0
    dotnetDurationMs: int = 0
    combineDurationMs: int = 0
    analysisDurationMs: int = 0
    predictionDurationMs: int = 0
    conclusionDurationMs: int = 0
    totalDurationMs: int = 0
    plannerDurationMs: Optional[float] = None  # ← added: present in durations_payload
    # Backward-compat camelCase fields
    confidence: Optional[float] = None
    queryType: Optional[str] = None
    operationCount: Optional[int] = None
    entityResolutionMs: Optional[float] = None
    planningMs: Optional[float] = None
    widgetMs: Optional[float] = None
    responseAssemblyMs: Optional[float] = None
    # Backward-compat snake_case fields
    planner_duration: Optional[float] = None
    intent_duration: Optional[float] = None
    dotnet_duration: Optional[float] = None
    combiner_duration: Optional[float] = None
    report_duration: Optional[float] = None
    total_duration: Optional[float] = None
    entity_resolution_ms: Optional[float] = None
    planning_ms: Optional[float] = None
    widget_ms: Optional[float] = None
    response_assembly_ms: Optional[float] = None


class CardItem(_BaseModel):
    title: str
    subtitle: str
    value: str
    description: str


class CardData(_BaseModel):
    cards: List[CardItem]


class TableColumn(_BaseModel):
    key: str
    label: str


class TableData(_BaseModel):
    columns: List[TableColumn]
    rows: List[Dict[str, Any]]


class BarChartData(_BaseModel):
    xKey: str
    yKey: str
    rows: List[Dict[str, Any]]


class SeriesItem(_BaseModel):
    key: str
    label: str


class LineChartData(_BaseModel):
    xKey: str
    series: List[SeriesItem]
    rows: List[Dict[str, Any]]


class PieChartItem(_BaseModel):
    label: str
    value: Any


class PieChartData(_BaseModel):
    rows: List[PieChartItem]


class FormattedData(_BaseModel):
    type: str
    title: str
    data: Optional[Dict[str, Any]] = None
    analysis: Dict[str, Any] = Field(default_factory=dict)
    prediction: Dict[str, Any] = Field(default_factory=dict)
    conclusion: Dict[str, Any] = Field(default_factory=dict)
    presentation: Optional[str] = None
    chart_type: Optional[str] = None
    comparison: Optional[bool] = None
    trend: Optional[bool] = None
    group_by: Optional[str] = None
    metric: Optional[str] = None


class WidgetItem(_BaseModel):
    """Single self-contained widget in a multi-widget response."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    title: str
    data: Dict[str, Any] = Field(default_factory=dict)


class FinalResponse(_BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    status: bool
    message: str
    # Comparison responses intentionally use a single widget dict, while
    # standard responses use a list of widget dicts.
    formattedData: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    analysis: str = ""
    prediction: str = ""
    conclusion: str = ""
    suggestedQuestions: List[str] = Field(default_factory=list)
    dotnetPayload: Any = None
    sessionId: str = ""
    queryType: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    overallConfidence: float = 0.0
    partialFailure: bool = False
    failedSections: List[Any] = Field(default_factory=list)


# Alias used throughout admin_pipeline.py
FinalResponseModel = FinalResponse
