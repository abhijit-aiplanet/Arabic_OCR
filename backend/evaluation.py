"""
OCR Evaluation Framework

Provides metrics and evaluation tools for measuring OCR accuracy:
- Field-level accuracy (exact match)
- Character Error Rate (CER)
- Blank detection accuracy
- Hallucination rate
- Confidence calibration

Directory Structure:
    evaluation/
    ├── ground_truth/
    │   ├── form_001.json  # Manual annotations
    │   └── ...
    ├── images/
    │   ├── form_001.png
    │   └── ...
    └── reports/
        └── evaluation_results.json
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import os
from datetime import datetime


# =============================================================================
# METRICS DATA STRUCTURES
# =============================================================================

@dataclass
class FieldMetrics:
    """Metrics for a single field."""
    field_name: str
    
    # Accuracy
    exact_match: bool = False
    cer: float = 0.0  # Character Error Rate
    
    # Classification
    predicted_empty: bool = False
    actual_empty: bool = False
    
    # Confidence
    confidence_score: float = 0.0
    confidence_level: str = ""
    
    # Hallucination
    is_hallucinated: bool = False
    hallucination_score: float = 0.0


@dataclass
class DocumentMetrics:
    """Metrics for a single document."""
    document_id: str
    image_path: str
    
    # Field-level metrics
    field_metrics: List[FieldMetrics] = field(default_factory=list)
    
    # Document-level aggregates
    total_fields: int = 0
    exact_match_count: int = 0
    average_cer: float = 0.0
    
    # Blank detection
    blank_precision: float = 0.0  # Of predicted blanks, how many were actually blank
    blank_recall: float = 0.0     # Of actual blanks, how many did we detect
    
    # Hallucination
    hallucination_count: int = 0
    hallucination_rate: float = 0.0
    
    # Processing
    processing_time_ms: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report across all documents."""
    # Summary
    total_documents: int = 0
    total_fields: int = 0
    
    # Accuracy metrics
    field_accuracy: float = 0.0       # Exact match rate
    average_cer: float = 0.0          # Average Character Error Rate
    
    # Blank detection
    blank_precision: float = 0.0
    blank_recall: float = 0.0
    blank_f1: float = 0.0
    
    # Hallucination
    hallucination_rate: float = 0.0   # % of outputs that don't exist in image
    
    # Confidence calibration
    confidence_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Per-document results
    document_results: List[DocumentMetrics] = field(default_factory=list)
    
    # Metadata
    evaluation_timestamp: str = ""
    model_version: str = ""
    
    # Warnings and notes
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# GROUND TRUTH FORMAT
# =============================================================================

@dataclass
class GroundTruthField:
    """Ground truth for a single field."""
    field_name: str
    value: str
    is_empty: bool = False
    is_handwritten: bool = True
    notes: Optional[str] = None


@dataclass
class GroundTruth:
    """Ground truth for a document."""
    document_id: str
    image_path: str
    fields: List[GroundTruthField] = field(default_factory=list)
    annotator: Optional[str] = None
    annotation_date: Optional[str] = None
    notes: Optional[str] = None


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_cer(predicted: str, actual: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    
    CER = (Substitutions + Deletions + Insertions) / Total Characters in Reference
    
    Returns:
        Float between 0.0 (perfect) and 1.0+ (very poor)
    """
    if not actual:
        return 0.0 if not predicted else 1.0
    
    if not predicted:
        return 1.0
    
    # Normalize for comparison
    pred_clean = predicted.strip()
    actual_clean = actual.strip()
    
    if pred_clean == actual_clean:
        return 0.0
    
    # Levenshtein distance
    distance = levenshtein_distance(pred_clean, actual_clean)
    cer = distance / len(actual_clean)
    
    return round(min(cer, 2.0), 4)  # Cap at 2.0 (200%)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_blank_detection_metrics(
    predicted_blanks: List[bool],
    actual_blanks: List[bool]
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 for blank detection.
    
    Returns:
        (precision, recall, f1)
    """
    if len(predicted_blanks) != len(actual_blanks):
        raise ValueError("Prediction and actual lists must have same length")
    
    true_positives = sum(1 for p, a in zip(predicted_blanks, actual_blanks) if p and a)
    false_positives = sum(1 for p, a in zip(predicted_blanks, actual_blanks) if p and not a)
    false_negatives = sum(1 for p, a in zip(predicted_blanks, actual_blanks) if not p and a)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return round(precision, 4), round(recall, 4), round(f1, 4)


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class OCREvaluator:
    """
    Evaluator for OCR outputs against ground truth.
    
    Usage:
        evaluator = OCREvaluator()
        
        # Add predictions and ground truth
        evaluator.add_result(document_id, predictions, ground_truth)
        
        # Generate report
        report = evaluator.generate_report()
    """
    
    # Markers indicating empty fields
    EMPTY_MARKERS = ["[فارغ]", "[EMPTY]", "[empty]", "فارغ", "", "-"]
    
    def __init__(self, model_version: str = "unknown"):
        self.model_version = model_version
        self.document_results: List[DocumentMetrics] = []
        self.all_field_metrics: List[FieldMetrics] = []
    
    def add_result(
        self,
        document_id: str,
        predictions: Dict[str, str],
        ground_truth: Dict[str, str],
        confidence_scores: Optional[Dict[str, float]] = None,
        confidence_levels: Optional[Dict[str, str]] = None,
        processing_time_ms: float = 0.0,
        image_path: str = ""
    ):
        """
        Add a document result for evaluation.
        
        Args:
            document_id: Unique identifier for the document
            predictions: Dict of field_name -> predicted_value
            ground_truth: Dict of field_name -> actual_value
            confidence_scores: Optional dict of field_name -> confidence score
            confidence_levels: Optional dict of field_name -> confidence level
            processing_time_ms: Processing time in milliseconds
            image_path: Path to the source image
        """
        confidence_scores = confidence_scores or {}
        confidence_levels = confidence_levels or {}
        
        field_metrics = []
        predicted_blanks = []
        actual_blanks = []
        exact_matches = 0
        cers = []
        hallucination_count = 0
        
        # Evaluate each field
        all_fields = set(predictions.keys()) | set(ground_truth.keys())
        
        for field_name in all_fields:
            pred = predictions.get(field_name, "")
            actual = ground_truth.get(field_name, "")
            
            # Determine if predicted/actual are empty
            pred_is_empty = self._is_empty(pred)
            actual_is_empty = self._is_empty(actual)
            
            predicted_blanks.append(pred_is_empty)
            actual_blanks.append(actual_is_empty)
            
            # Calculate exact match
            exact_match = self._normalize_for_comparison(pred) == self._normalize_for_comparison(actual)
            if exact_match:
                exact_matches += 1
            
            # Calculate CER (only for non-empty fields)
            if not actual_is_empty:
                cer = calculate_cer(pred, actual)
                cers.append(cer)
            else:
                cer = 0.0
            
            # Detect hallucination (predicted non-empty when actual is empty)
            is_hallucinated = not pred_is_empty and actual_is_empty
            if is_hallucinated:
                hallucination_count += 1
            
            field_metric = FieldMetrics(
                field_name=field_name,
                exact_match=exact_match,
                cer=cer,
                predicted_empty=pred_is_empty,
                actual_empty=actual_is_empty,
                confidence_score=confidence_scores.get(field_name, 0.0),
                confidence_level=confidence_levels.get(field_name, ""),
                is_hallucinated=is_hallucinated,
                hallucination_score=1.0 if is_hallucinated else 0.0
            )
            
            field_metrics.append(field_metric)
            self.all_field_metrics.append(field_metric)
        
        # Calculate document-level metrics
        blank_precision, blank_recall, _ = calculate_blank_detection_metrics(
            predicted_blanks, actual_blanks
        )
        
        total_fields = len(all_fields)
        avg_cer = sum(cers) / len(cers) if cers else 0.0
        hallucination_rate = hallucination_count / total_fields if total_fields > 0 else 0.0
        
        doc_metrics = DocumentMetrics(
            document_id=document_id,
            image_path=image_path,
            field_metrics=field_metrics,
            total_fields=total_fields,
            exact_match_count=exact_matches,
            average_cer=round(avg_cer, 4),
            blank_precision=blank_precision,
            blank_recall=blank_recall,
            hallucination_count=hallucination_count,
            hallucination_rate=round(hallucination_rate, 4),
            processing_time_ms=processing_time_ms
        )
        
        self.document_results.append(doc_metrics)
    
    def generate_report(self) -> EvaluationReport:
        """Generate the complete evaluation report."""
        if not self.document_results:
            return EvaluationReport(
                evaluation_timestamp=datetime.now().isoformat(),
                model_version=self.model_version,
                warnings=["No documents evaluated"]
            )
        
        total_documents = len(self.document_results)
        total_fields = sum(d.total_fields for d in self.document_results)
        
        # Field accuracy
        total_exact_matches = sum(d.exact_match_count for d in self.document_results)
        field_accuracy = total_exact_matches / total_fields if total_fields > 0 else 0.0
        
        # Average CER
        all_cers = [f.cer for f in self.all_field_metrics if not f.actual_empty]
        average_cer = sum(all_cers) / len(all_cers) if all_cers else 0.0
        
        # Blank detection
        all_predicted_blanks = [f.predicted_empty for f in self.all_field_metrics]
        all_actual_blanks = [f.actual_empty for f in self.all_field_metrics]
        blank_precision, blank_recall, blank_f1 = calculate_blank_detection_metrics(
            all_predicted_blanks, all_actual_blanks
        )
        
        # Hallucination rate
        total_hallucinations = sum(d.hallucination_count for d in self.document_results)
        hallucination_rate = total_hallucinations / total_fields if total_fields > 0 else 0.0
        
        # Confidence calibration
        confidence_accuracy = self._calculate_confidence_calibration()
        
        # Warnings
        warnings = []
        if hallucination_rate > 0.05:
            warnings.append(f"High hallucination rate: {hallucination_rate:.1%}")
        if blank_recall < 0.90:
            warnings.append(f"Low blank detection recall: {blank_recall:.1%}")
        if field_accuracy < 0.80:
            warnings.append(f"Low field accuracy: {field_accuracy:.1%}")
        
        return EvaluationReport(
            total_documents=total_documents,
            total_fields=total_fields,
            field_accuracy=round(field_accuracy, 4),
            average_cer=round(average_cer, 4),
            blank_precision=blank_precision,
            blank_recall=blank_recall,
            blank_f1=blank_f1,
            hallucination_rate=round(hallucination_rate, 4),
            confidence_accuracy=confidence_accuracy,
            document_results=self.document_results,
            evaluation_timestamp=datetime.now().isoformat(),
            model_version=self.model_version,
            warnings=warnings
        )
    
    def _is_empty(self, value: str) -> bool:
        """Check if a value represents an empty field."""
        if not value:
            return True
        clean = value.strip()
        return clean in self.EMPTY_MARKERS or clean == ""
    
    def _normalize_for_comparison(self, value: str) -> str:
        """Normalize a value for comparison."""
        if self._is_empty(value):
            return ""
        
        # Convert Arabic digits to Western for comparison
        arabic_digits = '٠١٢٣٤٥٦٧٨٩'
        western_digits = '0123456789'
        result = value.strip()
        for ar, we in zip(arabic_digits, western_digits):
            result = result.replace(ar, we)
        
        # Remove extra whitespace
        result = ' '.join(result.split())
        
        return result
    
    def _calculate_confidence_calibration(self) -> Dict[str, float]:
        """
        Calculate accuracy at each confidence level.
        
        Returns dict like:
            {"high": 0.95, "medium": 0.78, "low": 0.45}
        """
        calibration: Dict[str, List[bool]] = {}
        
        for field in self.all_field_metrics:
            level = field.confidence_level or "unknown"
            if level not in calibration:
                calibration[level] = []
            calibration[level].append(field.exact_match)
        
        result = {}
        for level, matches in calibration.items():
            if matches:
                result[level] = round(sum(matches) / len(matches), 4)
        
        return result


# =============================================================================
# REPORT GENERATION
# =============================================================================

def save_report(report: EvaluationReport, output_path: str):
    """Save evaluation report to JSON file."""
    report_dict = {
        "summary": {
            "total_documents": report.total_documents,
            "total_fields": report.total_fields,
            "field_accuracy": report.field_accuracy,
            "average_cer": report.average_cer,
            "blank_precision": report.blank_precision,
            "blank_recall": report.blank_recall,
            "blank_f1": report.blank_f1,
            "hallucination_rate": report.hallucination_rate,
        },
        "confidence_calibration": report.confidence_accuracy,
        "warnings": report.warnings,
        "metadata": {
            "evaluation_timestamp": report.evaluation_timestamp,
            "model_version": report.model_version,
        },
        "documents": [
            {
                "document_id": d.document_id,
                "image_path": d.image_path,
                "total_fields": d.total_fields,
                "exact_match_count": d.exact_match_count,
                "average_cer": d.average_cer,
                "hallucination_rate": d.hallucination_rate,
                "processing_time_ms": d.processing_time_ms,
            }
            for d in report.document_results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    """Load ground truth from a JSON file."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Support both flat format and nested format
    if "fields" in data:
        return {f["field_name"]: f["value"] for f in data["fields"]}
    else:
        return data


# =============================================================================
# DIRECTORY-BASED EVALUATION
# =============================================================================

def evaluate_directory(
    predictions_dir: str,
    ground_truth_dir: str,
    output_path: str,
    model_version: str = "unknown"
) -> EvaluationReport:
    """
    Evaluate all predictions against ground truth in directories.
    
    Expected structure:
        predictions_dir/
            form_001.json  # {"field_name": "value", ...}
            form_002.json
        ground_truth_dir/
            form_001.json  # {"field_name": "value", ...}
            form_002.json
    """
    evaluator = OCREvaluator(model_version=model_version)
    
    pred_files = list(Path(predictions_dir).glob("*.json"))
    
    for pred_file in pred_files:
        gt_file = Path(ground_truth_dir) / pred_file.name
        
        if not gt_file.exists():
            print(f"Warning: No ground truth for {pred_file.name}")
            continue
        
        # Load predictions
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Load ground truth
        ground_truth = load_ground_truth(str(gt_file))
        
        # Add to evaluator
        evaluator.add_result(
            document_id=pred_file.stem,
            predictions=predictions,
            ground_truth=ground_truth
        )
    
    # Generate and save report
    report = evaluator.generate_report()
    save_report(report, output_path)
    
    return report


# =============================================================================
# QUICK EVALUATION FUNCTION
# =============================================================================

def quick_evaluate(
    predicted: Dict[str, str],
    actual: Dict[str, str]
) -> Dict[str, Any]:
    """
    Quick evaluation of a single prediction against ground truth.
    
    Returns:
        Dict with accuracy metrics
    """
    evaluator = OCREvaluator()
    evaluator.add_result("quick_eval", predicted, actual)
    report = evaluator.generate_report()
    
    return {
        "field_accuracy": report.field_accuracy,
        "average_cer": report.average_cer,
        "hallucination_rate": report.hallucination_rate,
        "blank_precision": report.blank_precision,
        "blank_recall": report.blank_recall,
    }
