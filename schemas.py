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
    raw_query: Optional[str] = None
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
    group_by: Optional[str] = None
    sort_by: Optional[str] = None
    aggregation: Optional[str] = None
    top_n: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None
    widget_hint: Optional[str] = None
    widgetHint: Optional[str] = None


class DotNetPayloadModel(BaseModel):
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

    success: Optional[bool] = None
    commandLabel: Optional[str] = None
    status: Optional[bool] = None
    data: Optional[Any] = None
    message: Optional[str] = None
    records: Optional[List[Dict[str, Any]]] = None

class CombinedResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insights: List[str] = Field(default_factory=list)
    summary: str = ""
    statistics: Dict[str, Any] = Field(default_factory=dict)
    observations: List[str] = Field(default_factory=list)
    predictions: List[str] = Field(default_factory=list)

class PredictionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend: str = ""
    projection: Optional[str] = None
    heuristicEstimate: Optional[str] = None
    shortTerm: Optional[str] = None
    futureTrends: List[str] = Field(default_factory=list)

class ConclusionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    bullets: List[str] = Field(default_factory=list)

class SuggestedQuestionModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str

class WidgetModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    section: str = ""
    widgetType: str = ""
    type: Optional[str] = None  # For backward compatibility

class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requestId: str = ""
    traceId: str = ""
    sessionId: str = ""
    timings: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
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
    entityResolutionMs: Optional[float] = None
    planningMs: Optional[float] = None
    widgetMs: Optional[float] = None
    responseAssemblyMs: Optional[float] = None
    entity_resolution_ms: Optional[float] = None
    planning_ms: Optional[float] = None
    widget_ms: Optional[float] = None
    response_assembly_ms: Optional[float] = None

class CardItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    subtitle: str
    value: str
    description: str

class CardData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cards: List[CardItem]

class TableColumn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    key: str
    label: str

class TableData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    columns: List[TableColumn]
    rows: List[Dict[str, Any]]

class BarChartData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    xKey: str
    yKey: str
    rows: List[Dict[str, Any]]

class SeriesItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    key: str
    label: str

class LineChartData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    xKey: str
    series: List[SeriesItem]
    rows: List[Dict[str, Any]]

class PieChartItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str
    value: Any

class PieChartData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rows: List[PieChartItem]

class FormattedData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    title: str
    data: Dict[str, Any]
    presentation: Optional[str] = None
    chart_type: Optional[str] = None
    comparison: Optional[bool] = None
    trend: Optional[bool] = None
    group_by: Optional[str] = None
    metric: Optional[str] = None

class FinalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: bool
    sessionId: str
    message: str
    queryType: str
    introMessage: str
    answer: Dict[str, Any]
    result: Dict[str, Any]
    widgets: List[Dict[str, Any]]
    widget: Optional[str] = None
    records: List[Dict[str, Any]] = Field(default_factory=list)
    analysis: Optional[Any] = None
    prediction: Optional[Any] = None
    conclusion: Optional[Any] = None
    intent: Dict[str, Any] = Field(default_factory=dict)
    formattedData: Optional[Dict[str, Any]] = None
    suggestedQuestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    overallConfidence: float = 0.0
    partialFailure: bool = False
    failedSections: List[str] = Field(default_factory=list)

FinalResponseModel = FinalResponse
