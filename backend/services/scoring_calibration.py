"""
Scoring Calibration System
Aligns AI-generated evaluation scores with human judgment through statistical calibration.
Part of Task 6.3 - Implement Scoring Calibration System.
"""

import asyncio
import json
import logging
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from pathlib import Path

from services.evaluator_models import EvaluationCriteria, EvaluationResult

logger = logging.getLogger(__name__)

class CalibrationMethod(str, Enum):
    """Available calibration methods."""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    ISOTONIC_REGRESSION = "isotonic_regression"
    BETA_CALIBRATION = "beta_calibration"
    PLATT_SCALING = "platt_scaling"

@dataclass
class HumanScore:
    """Human evaluation score for calibration."""
    trace_id: str
    evaluator_email: str
    criteria: EvaluationCriteria
    score: float  # 0.0 to 1.0
    reasoning: str
    confidence: float  # 0.0 to 1.0
    evaluation_time_ms: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CalibrationDataPoint:
    """Single data point for calibration."""
    trace_id: str
    criteria: EvaluationCriteria
    ai_score: float
    human_score: float
    evaluator_model: str
    human_evaluator: str
    ai_confidence: float
    human_confidence: float
    created_at: str

@dataclass
class CalibrationModel:
    """Calibration model for a specific criteria and evaluator."""
    criteria: EvaluationCriteria
    evaluator_model: str
    method: CalibrationMethod
    model_data: bytes  # Pickled sklearn model
    training_data_count: int
    last_trained: str
    performance_metrics: Dict[str, float]
    version: str = "1.0"

@dataclass
class CalibrationResult:
    """Result of score calibration."""
    original_score: float
    calibrated_score: float
    confidence_adjustment: float
    calibration_method: CalibrationMethod
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ScoringCalibrationSystem:
    """System for calibrating AI evaluation scores with human judgment."""
    
    def __init__(self, data_dir: str = "data/calibration"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for development (would use database in production)
        self.human_scores: List[HumanScore] = []
        self.calibration_models: Dict[str, CalibrationModel] = {}
        self.calibration_data: List[CalibrationDataPoint] = []
        
        # Load existing data
        self._load_data()
        
        # Calibration settings
        self.min_training_samples = 20
        self.recalibration_threshold = 50  # New samples before recalibration
        self.confidence_threshold = 0.7
        
    def _load_data(self):
        """Load existing calibration data from disk."""
        try:
            # Load human scores
            human_scores_file = self.data_dir / "human_scores.json"
            if human_scores_file.exists():
                with open(human_scores_file, 'r') as f:
                    data = json.load(f)
                    self.human_scores = [HumanScore(**item) for item in data]
            
            # Load calibration data
            calibration_data_file = self.data_dir / "calibration_data.json"
            if calibration_data_file.exists():
                with open(calibration_data_file, 'r') as f:
                    data = json.load(f)
                    self.calibration_data = [CalibrationDataPoint(**item) for item in data]
            
            # Load calibration models
            models_dir = self.data_dir / "models"
            if models_dir.exists():
                for model_file in models_dir.glob("*.json"):
                    with open(model_file, 'r') as f:
                        model_info = json.load(f)
                    
                    # Load the actual model data
                    model_data_file = models_dir / f"{model_file.stem}.pkl"
                    if model_data_file.exists():
                        with open(model_data_file, 'rb') as f:
                            model_data = f.read()
                        
                        model = CalibrationModel(
                            criteria=EvaluationCriteria(model_info['criteria']),
                            evaluator_model=model_info['evaluator_model'],
                            method=CalibrationMethod(model_info['method']),
                            model_data=model_data,
                            training_data_count=model_info['training_data_count'],
                            last_trained=model_info['last_trained'],
                            performance_metrics=model_info['performance_metrics'],
                            version=model_info.get('version', '1.0')
                        )
                        
                        key = f"{model.criteria.value}_{model.evaluator_model}"
                        self.calibration_models[key] = model
                        
        except Exception as e:
            logger.warning(f"Error loading calibration data: {e}")
    
    def _save_data(self):
        """Save calibration data to disk."""
        try:
            # Save human scores
            human_scores_file = self.data_dir / "human_scores.json"
            with open(human_scores_file, 'w') as f:
                data = [
                    {
                        "trace_id": score.trace_id,
                        "evaluator_email": score.evaluator_email,
                        "criteria": score.criteria.value,
                        "score": score.score,
                        "reasoning": score.reasoning,
                        "confidence": score.confidence,
                        "evaluation_time_ms": score.evaluation_time_ms,
                        "created_at": score.created_at,
                        "metadata": score.metadata
                    }
                    for score in self.human_scores
                ]
                json.dump(data, f, indent=2)
            
            # Save calibration data
            calibration_data_file = self.data_dir / "calibration_data.json"
            with open(calibration_data_file, 'w') as f:
                data = [
                    {
                        "trace_id": point.trace_id,
                        "criteria": point.criteria.value,
                        "ai_score": point.ai_score,
                        "human_score": point.human_score,
                        "evaluator_model": point.evaluator_model,
                        "human_evaluator": point.human_evaluator,
                        "ai_confidence": point.ai_confidence,
                        "human_confidence": point.human_confidence,
                        "created_at": point.created_at
                    }
                    for point in self.calibration_data
                ]
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
    
    async def add_human_score(self, human_score: HumanScore) -> bool:
        """Add a human evaluation score for calibration."""
        try:
            self.human_scores.append(human_score)
            self._save_data()
            
            # Check if we can create a calibration data point
            await self._try_create_calibration_point(human_score)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding human score: {e}")
            return False
    
    async def _try_create_calibration_point(self, human_score: HumanScore):
        """Try to create a calibration data point by finding matching AI evaluation."""
        try:
            # In a real system, this would query the database for AI evaluations
            # For now, we'll simulate finding a matching AI evaluation
            
            # This would be replaced with actual database query
            # ai_evaluation = await self._find_ai_evaluation(human_score.trace_id, human_score.criteria)
            
            # Simulated AI evaluation for demonstration
            ai_evaluation = None  # Would be actual EvaluationResult
            
            if ai_evaluation:
                calibration_point = CalibrationDataPoint(
                    trace_id=human_score.trace_id,
                    criteria=human_score.criteria,
                    ai_score=ai_evaluation.score,
                    human_score=human_score.score,
                    evaluator_model=ai_evaluation.evaluator_model,
                    human_evaluator=human_score.evaluator_email,
                    ai_confidence=ai_evaluation.confidence,
                    human_confidence=human_score.confidence,
                    created_at=datetime.utcnow().isoformat()
                )
                
                self.calibration_data.append(calibration_point)
                self._save_data()
                
                # Check if we need to retrain calibration models
                await self._check_retrain_models(human_score.criteria, ai_evaluation.evaluator_model)
                
        except Exception as e:
            logger.error(f"Error creating calibration point: {e}")
    
    async def _check_retrain_models(self, criteria: EvaluationCriteria, evaluator_model: str):
        """Check if calibration models need retraining."""
        try:
            key = f"{criteria.value}_{evaluator_model}"
            
            # Count new data points since last training
            if key in self.calibration_models:
                last_trained = datetime.fromisoformat(self.calibration_models[key].last_trained)
                new_points = [
                    point for point in self.calibration_data
                    if (point.criteria == criteria and 
                        point.evaluator_model == evaluator_model and
                        datetime.fromisoformat(point.created_at) > last_trained)
                ]
                
                if len(new_points) >= self.recalibration_threshold:
                    await self.train_calibration_model(criteria, evaluator_model)
            else:
                # No existing model, check if we have enough data to train
                relevant_points = [
                    point for point in self.calibration_data
                    if point.criteria == criteria and point.evaluator_model == evaluator_model
                ]
                
                if len(relevant_points) >= self.min_training_samples:
                    await self.train_calibration_model(criteria, evaluator_model)
                    
        except Exception as e:
            logger.error(f"Error checking retrain models: {e}")
    
    async def train_calibration_model(
        self, 
        criteria: EvaluationCriteria, 
        evaluator_model: str,
        method: CalibrationMethod = CalibrationMethod.ISOTONIC_REGRESSION
    ) -> bool:
        """Train a calibration model for specific criteria and evaluator."""
        try:
            # Get training data
            training_data = [
                point for point in self.calibration_data
                if point.criteria == criteria and point.evaluator_model == evaluator_model
            ]
            
            if len(training_data) < self.min_training_samples:
                logger.warning(f"Insufficient training data for {criteria.value}_{evaluator_model}: {len(training_data)} samples")
                return False
            
            # Prepare data for training
            ai_scores = np.array([point.ai_score for point in training_data])
            human_scores = np.array([point.human_score for point in training_data])
            
            # Train calibration model
            model, metrics = await self._train_model(ai_scores, human_scores, method)
            
            if model is None:
                return False
            
            # Serialize model
            model_data = pickle.dumps(model)
            
            # Create calibration model
            calibration_model = CalibrationModel(
                criteria=criteria,
                evaluator_model=evaluator_model,
                method=method,
                model_data=model_data,
                training_data_count=len(training_data),
                last_trained=datetime.utcnow().isoformat(),
                performance_metrics=metrics
            )
            
            # Store model
            key = f"{criteria.value}_{evaluator_model}"
            self.calibration_models[key] = calibration_model
            
            # Save model to disk
            await self._save_model(calibration_model)
            
            logger.info(f"Trained calibration model for {key} with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training calibration model: {e}")
            return False
    
    async def _train_model(
        self, 
        ai_scores: np.ndarray, 
        human_scores: np.ndarray, 
        method: CalibrationMethod
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """Train the actual calibration model."""
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.model_selection import cross_val_score
            
            # Reshape for sklearn
            X = ai_scores.reshape(-1, 1)
            y = human_scores
            
            # Train model based on method
            if method == CalibrationMethod.LINEAR_REGRESSION:
                model = LinearRegression()
                model.fit(X, y)
                
            elif method == CalibrationMethod.POLYNOMIAL_REGRESSION:
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression())
                ])
                model.fit(X, y)
                
            elif method == CalibrationMethod.ISOTONIC_REGRESSION:
                model = IsotonicRegression(out_of_bounds='clip')
                model.fit(ai_scores, human_scores)
                
            else:
                # Default to isotonic regression
                model = IsotonicRegression(out_of_bounds='clip')
                model.fit(ai_scores, human_scores)
            
            # Calculate performance metrics
            predictions = model.predict(X if method != CalibrationMethod.ISOTONIC_REGRESSION else ai_scores)
            
            metrics = {
                'mse': float(mean_squared_error(y, predictions)),
                'mae': float(mean_absolute_error(y, predictions)),
                'r2': float(r2_score(y, predictions)),
                'training_samples': len(ai_scores)
            }
            
            # Add cross-validation score if enough data
            if len(ai_scores) >= 10:
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(ai_scores)//2), scoring='neg_mean_squared_error')
                metrics['cv_mse'] = float(-cv_scores.mean())
                metrics['cv_std'] = float(cv_scores.std())
            
            return model, metrics
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple linear calibration")
            # Fallback to simple linear calibration
            return await self._simple_linear_calibration(ai_scores, human_scores)
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, {}
    
    async def _simple_linear_calibration(
        self, 
        ai_scores: np.ndarray, 
        human_scores: np.ndarray
    ) -> Tuple[Optional[Dict], Dict[str, float]]:
        """Simple linear calibration fallback when sklearn is not available."""
        try:
            # Calculate linear regression manually
            n = len(ai_scores)
            sum_x = np.sum(ai_scores)
            sum_y = np.sum(human_scores)
            sum_xy = np.sum(ai_scores * human_scores)
            sum_x2 = np.sum(ai_scores ** 2)
            
            # Calculate slope and intercept
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            intercept = (sum_y - slope * sum_x) / n
            
            # Create simple model dict
            model = {
                'type': 'linear',
                'slope': float(slope),
                'intercept': float(intercept)
            }
            
            # Calculate metrics
            predictions = slope * ai_scores + intercept
            mse = np.mean((human_scores - predictions) ** 2)
            mae = np.mean(np.abs(human_scores - predictions))
            
            # Calculate R²
            ss_res = np.sum((human_scores - predictions) ** 2)
            ss_tot = np.sum((human_scores - np.mean(human_scores)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'training_samples': len(ai_scores),
                'method': 'simple_linear'
            }
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error in simple linear calibration: {e}")
            return None, {}
    
    async def _save_model(self, calibration_model: CalibrationModel):
        """Save calibration model to disk."""
        try:
            models_dir = self.data_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            key = f"{calibration_model.criteria.value}_{calibration_model.evaluator_model}"
            
            # Save model metadata
            model_info = {
                'criteria': calibration_model.criteria.value,
                'evaluator_model': calibration_model.evaluator_model,
                'method': calibration_model.method.value,
                'training_data_count': calibration_model.training_data_count,
                'last_trained': calibration_model.last_trained,
                'performance_metrics': calibration_model.performance_metrics,
                'version': calibration_model.version
            }
            
            with open(models_dir / f"{key}.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Save model data
            with open(models_dir / f"{key}.pkl", 'wb') as f:
                f.write(calibration_model.model_data)
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    async def calibrate_score(
        self, 
        ai_score: float, 
        criteria: EvaluationCriteria, 
        evaluator_model: str,
        confidence: float = 0.8
    ) -> CalibrationResult:
        """Calibrate an AI-generated score using trained models."""
        try:
            key = f"{criteria.value}_{evaluator_model}"
            
            if key not in self.calibration_models:
                # No calibration model available, return original score
                return CalibrationResult(
                    original_score=ai_score,
                    calibrated_score=ai_score,
                    confidence_adjustment=0.0,
                    calibration_method=CalibrationMethod.LINEAR_REGRESSION,
                    model_version="none",
                    metadata={"reason": "no_calibration_model"}
                )
            
            calibration_model = self.calibration_models[key]
            
            # Load and apply calibration model
            try:
                model = pickle.loads(calibration_model.model_data)
                
                if calibration_model.method == CalibrationMethod.ISOTONIC_REGRESSION:
                    calibrated_score = model.predict([ai_score])[0]
                else:
                    calibrated_score = model.predict([[ai_score]])[0]
                    
            except:
                # Fallback to simple linear calibration
                model_data = pickle.loads(calibration_model.model_data)
                if isinstance(model_data, dict) and model_data.get('type') == 'linear':
                    calibrated_score = model_data['slope'] * ai_score + model_data['intercept']
                else:
                    calibrated_score = ai_score
            
            # Ensure score is in valid range
            calibrated_score = max(0.0, min(1.0, calibrated_score))
            
            # Calculate confidence adjustment based on model performance
            confidence_adjustment = self._calculate_confidence_adjustment(
                calibration_model.performance_metrics, confidence
            )
            
            return CalibrationResult(
                original_score=ai_score,
                calibrated_score=calibrated_score,
                confidence_adjustment=confidence_adjustment,
                calibration_method=calibration_model.method,
                model_version=calibration_model.version,
                metadata={
                    "training_samples": calibration_model.training_data_count,
                    "model_performance": calibration_model.performance_metrics,
                    "last_trained": calibration_model.last_trained
                }
            )
            
        except Exception as e:
            logger.error(f"Error calibrating score: {e}")
            # Return original score on error
            return CalibrationResult(
                original_score=ai_score,
                calibrated_score=ai_score,
                confidence_adjustment=0.0,
                calibration_method=CalibrationMethod.LINEAR_REGRESSION,
                model_version="error",
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence_adjustment(
        self, 
        performance_metrics: Dict[str, float], 
        original_confidence: float
    ) -> float:
        """Calculate confidence adjustment based on model performance."""
        try:
            # Use R² score to adjust confidence
            r2 = performance_metrics.get('r2', 0.5)
            mse = performance_metrics.get('mse', 0.1)
            
            # Higher R² and lower MSE should increase confidence
            r2_factor = max(0.0, min(1.0, r2))
            mse_factor = max(0.0, min(1.0, 1.0 - mse))
            
            # Calculate adjustment
            performance_factor = (r2_factor + mse_factor) / 2.0
            adjustment = (performance_factor - 0.5) * 0.2  # Max ±0.1 adjustment
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating confidence adjustment: {e}")
            return 0.0
    
    async def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration system statistics."""
        try:
            stats = {
                "total_human_scores": len(self.human_scores),
                "total_calibration_points": len(self.calibration_data),
                "trained_models": len(self.calibration_models),
                "models_by_criteria": {},
                "recent_activity": {},
                "performance_summary": {}
            }
            
            # Group by criteria
            for criteria in EvaluationCriteria:
                criteria_points = [p for p in self.calibration_data if p.criteria == criteria]
                criteria_models = [m for m in self.calibration_models.values() if m.criteria == criteria]
                
                stats["models_by_criteria"][criteria.value] = {
                    "calibration_points": len(criteria_points),
                    "trained_models": len(criteria_models),
                    "evaluator_models": list(set(m.evaluator_model for m in criteria_models))
                }
            
            # Recent activity (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_scores = [
                s for s in self.human_scores 
                if datetime.fromisoformat(s.created_at) > week_ago
            ]
            recent_points = [
                p for p in self.calibration_data 
                if datetime.fromisoformat(p.created_at) > week_ago
            ]
            
            stats["recent_activity"] = {
                "human_scores_last_week": len(recent_scores),
                "calibration_points_last_week": len(recent_points)
            }
            
            # Performance summary
            if self.calibration_models:
                r2_scores = [m.performance_metrics.get('r2', 0) for m in self.calibration_models.values()]
                mse_scores = [m.performance_metrics.get('mse', 0) for m in self.calibration_models.values()]
                
                stats["performance_summary"] = {
                    "average_r2": statistics.mean(r2_scores) if r2_scores else 0,
                    "average_mse": statistics.mean(mse_scores) if mse_scores else 0,
                    "best_r2": max(r2_scores) if r2_scores else 0,
                    "worst_mse": max(mse_scores) if mse_scores else 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting calibration stats: {e}")
            return {"error": str(e)}

# Global calibration system instance
calibration_system = ScoringCalibrationSystem() 