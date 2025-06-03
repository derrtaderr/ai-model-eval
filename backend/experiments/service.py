"""
A/B Testing Service
Comprehensive service for experiment management, traffic allocation, and statistical analysis.
"""

import logging
import hashlib
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_
from sqlalchemy.orm import selectinload
import numpy as np
from scipy import stats
import math

from .models import (
    Experiment, ExperimentVariant, ParticipantAssignment, ExperimentEvent, ExperimentResult,
    ExperimentStatus, VariantType, AllocationMethod, StatisticalTest
)
from services.cache_service import cache_service

logger = logging.getLogger(__name__)


class TrafficAllocator:
    """Handles traffic allocation for A/B tests."""
    
    def __init__(self):
        self.salt = "experiment_allocation_salt_2024"
    
    def generate_participant_hash(
        self, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None, 
        experiment_id: str = None
    ) -> str:
        """Generate stable hash for participant assignment."""
        # Use user_id if available, otherwise use session_id
        identifier = user_id if user_id else session_id
        if not identifier:
            raise ValueError("Either user_id or session_id must be provided")
        
        # Create stable hash combining identifier, experiment ID, and salt
        hash_input = f"{identifier}:{experiment_id}:{self.salt}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def should_include_in_experiment(
        self, 
        traffic_percentage: float, 
        participant_hash: str
    ) -> bool:
        """Determine if participant should be included in experiment."""
        # Use hash to get consistent allocation
        hash_int = int(participant_hash[:8], 16)  # Use first 8 chars as hex
        allocation_value = (hash_int % 100) + 1  # Convert to 1-100 range
        
        return allocation_value <= traffic_percentage
    
    def allocate_variant(
        self, 
        variants: List[Dict[str, Any]], 
        participant_hash: str
    ) -> str:
        """Allocate participant to a variant based on traffic weights."""
        if not variants:
            raise ValueError("No variants provided")
        
        # Normalize weights to sum to 100
        total_weight = sum(v['traffic_weight'] for v in variants)
        if total_weight == 0:
            raise ValueError("Total traffic weight cannot be zero")
        
        # Use hash for consistent allocation
        hash_int = int(participant_hash[8:16], 16)  # Use different part of hash
        allocation_value = hash_int % 10000  # 0-9999 range for precision
        
        # Calculate cumulative weights
        cumulative = 0
        target = (allocation_value / 10000) * total_weight
        
        for variant in variants:
            cumulative += variant['traffic_weight']
            if target <= cumulative:
                return variant['id']
        
        # Fallback to first variant
        return variants[0]['id']


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B test results."""
    
    def calculate_sample_size(
        self, 
        baseline_rate: float, 
        minimum_detectable_effect: float, 
        alpha: float = 0.05, 
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for experiment."""
        try:
            # Z-scores for alpha and power
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            
            # Expected rates
            p1 = baseline_rate
            p2 = baseline_rate * (1 + minimum_detectable_effect)
            
            # Pooled probability
            p_pooled = (p1 + p2) / 2
            
            # Sample size calculation
            numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                        z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            denominator = (p2 - p1) ** 2
            
            sample_size_per_group = math.ceil(numerator / denominator)
            
            return sample_size_per_group * 2  # Total for both groups
            
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return 1000  # Default fallback
    
    def perform_t_test(
        self, 
        control_values: List[float], 
        treatment_values: List[float], 
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Perform two-sample t-test."""
        if len(control_values) < 2 or len(treatment_values) < 2:
            return {
                "error": "Insufficient data for t-test",
                "p_value": None,
                "is_significant": False
            }
        
        try:
            # Perform Welch's t-test (unequal variances)
            statistic, p_value = stats.ttest_ind(
                treatment_values, 
                control_values, 
                equal_var=False
            )
            
            # Calculate confidence interval
            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)
            control_std = np.std(control_values, ddof=1)
            treatment_std = np.std(treatment_values, ddof=1)
            
            n1, n2 = len(control_values), len(treatment_values)
            
            # Standard error of difference
            se_diff = math.sqrt((control_std**2 / n1) + (treatment_std**2 / n2))
            
            # Degrees of freedom (Welch-Satterthwaite equation)
            df = ((control_std**2 / n1) + (treatment_std**2 / n2))**2 / (
                (control_std**2 / n1)**2 / (n1 - 1) + 
                (treatment_std**2 / n2)**2 / (n2 - 1)
            )
            
            # Critical value
            alpha = 1 - confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            # Confidence interval for difference
            diff = treatment_mean - control_mean
            margin_error = t_crit * se_diff
            
            # Effect size (Cohen's d)
            pooled_std = math.sqrt(((n1-1)*control_std**2 + (n2-1)*treatment_std**2) / (n1+n2-2))
            cohens_d = diff / pooled_std if pooled_std > 0 else 0
            
            return {
                "test_statistic": float(statistic),
                "p_value": float(p_value),
                "is_significant": p_value < (1 - confidence_level),
                "degrees_of_freedom": float(df),
                "control_mean": float(control_mean),
                "treatment_mean": float(treatment_mean),
                "control_std": float(control_std),
                "treatment_std": float(treatment_std),
                "absolute_effect": float(diff),
                "relative_effect": float((diff / control_mean * 100) if control_mean != 0 else 0),
                "confidence_interval_lower": float(diff - margin_error),
                "confidence_interval_upper": float(diff + margin_error),
                "effect_size_cohens_d": float(cohens_d),
                "pooled_variance": float(pooled_std**2)
            }
            
        except Exception as e:
            logger.error(f"Error performing t-test: {e}")
            return {
                "error": str(e),
                "p_value": None,
                "is_significant": False
            }
    
    def perform_chi_square_test(
        self, 
        control_successes: int, 
        control_total: int, 
        treatment_successes: int, 
        treatment_total: int,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Perform chi-square test for conversion rates."""
        try:
            # Create contingency table
            observed = np.array([
                [control_successes, control_total - control_successes],
                [treatment_successes, treatment_total - treatment_successes]
            ])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            
            # Calculate rates
            control_rate = control_successes / control_total if control_total > 0 else 0
            treatment_rate = treatment_successes / treatment_total if treatment_total > 0 else 0
            
            # Confidence interval for difference in proportions
            diff = treatment_rate - control_rate
            se_diff = math.sqrt(
                (control_rate * (1 - control_rate) / control_total) +
                (treatment_rate * (1 - treatment_rate) / treatment_total)
            )
            
            alpha = 1 - confidence_level
            z_crit = stats.norm.ppf(1 - alpha/2)
            margin_error = z_crit * se_diff
            
            return {
                "test_statistic": float(chi2),
                "p_value": float(p_value),
                "is_significant": p_value < (1 - confidence_level),
                "degrees_of_freedom": int(dof),
                "control_rate": float(control_rate),
                "treatment_rate": float(treatment_rate),
                "absolute_effect": float(diff),
                "relative_effect": float((diff / control_rate * 100) if control_rate != 0 else 0),
                "confidence_interval_lower": float(diff - margin_error),
                "confidence_interval_upper": float(diff + margin_error),
                "control_sample_size": control_total,
                "treatment_sample_size": treatment_total
            }
            
        except Exception as e:
            logger.error(f"Error performing chi-square test: {e}")
            return {
                "error": str(e),
                "p_value": None,
                "is_significant": False
            }


class ExperimentService:
    """Main service for A/B testing functionality."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.allocator = TrafficAllocator()
        self.analyzer = StatisticalAnalyzer()
    
    async def create_experiment(
        self, 
        team_id: str, 
        user_id: str, 
        experiment_data: Dict[str, Any]
    ) -> Experiment:
        """Create a new experiment."""
        try:
            # Calculate sample size if baseline data provided
            required_sample_size = None
            if experiment_data.get('baseline_conversion_rate'):
                required_sample_size = self.analyzer.calculate_sample_size(
                    baseline_rate=experiment_data['baseline_conversion_rate'],
                    minimum_detectable_effect=experiment_data.get('minimum_detectable_effect', 0.05),
                    alpha=1 - experiment_data.get('confidence_level', 0.95),
                    power=experiment_data.get('power', 0.8)
                )
            
            experiment = Experiment(
                team_id=team_id,
                created_by=user_id,
                name=experiment_data['name'],
                description=experiment_data.get('description'),
                hypothesis=experiment_data.get('hypothesis'),
                primary_metric=experiment_data['primary_metric'],
                secondary_metrics=experiment_data.get('secondary_metrics', []),
                traffic_percentage=experiment_data.get('traffic_percentage', 100.0),
                duration_days=experiment_data.get('duration_days'),
                confidence_level=experiment_data.get('confidence_level', 0.95),
                minimum_detectable_effect=experiment_data.get('minimum_detectable_effect', 0.05),
                statistical_test=experiment_data.get('statistical_test', StatisticalTest.T_TEST),
                targeting_rules=experiment_data.get('targeting_rules'),
                exclusion_rules=experiment_data.get('exclusion_rules'),
                required_sample_size=required_sample_size
            )
            
            self.session.add(experiment)
            await self.session.commit()
            await self.session.refresh(experiment)
            
            # Invalidate experiment cache
            await cache_service.delete_pattern(f"*{team_id}*", prefix="experiments")
            
            logger.info(f"Created experiment {experiment.id} for team {team_id}")
            return experiment
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating experiment: {e}")
            raise
    
    async def add_variant(
        self, 
        experiment_id: str, 
        variant_data: Dict[str, Any]
    ) -> ExperimentVariant:
        """Add a variant to an experiment."""
        try:
            variant = ExperimentVariant(
                experiment_id=experiment_id,
                name=variant_data['name'],
                description=variant_data.get('description'),
                variant_type=variant_data.get('variant_type', VariantType.TREATMENT),
                traffic_weight=variant_data['traffic_weight'],
                configuration=variant_data['configuration'],
                model_name=variant_data.get('model_name'),
                system_prompt=variant_data.get('system_prompt'),
                temperature=variant_data.get('temperature'),
                max_tokens=variant_data.get('max_tokens'),
                model_parameters=variant_data.get('model_parameters')
            )
            
            self.session.add(variant)
            await self.session.commit()
            await self.session.refresh(variant)
            
            logger.info(f"Added variant {variant.id} to experiment {experiment_id}")
            return variant
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error adding variant: {e}")
            raise
    
    async def assign_participant(
        self, 
        experiment_id: str, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Assign participant to experiment variant."""
        try:
            # Get experiment with variants
            stmt = select(Experiment).options(
                selectinload(Experiment.variants)
            ).where(
                and_(
                    Experiment.id == experiment_id,
                    Experiment.status == ExperimentStatus.ACTIVE
                )
            )
            
            result = await self.session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment or not experiment.variants:
                return None
            
            # Generate participant hash
            participant_hash = self.allocator.generate_participant_hash(
                user_id=user_id,
                session_id=session_id,
                experiment_id=experiment_id
            )
            
            # Check if already assigned
            existing_assignment = await self._get_existing_assignment(
                experiment_id, participant_hash
            )
            if existing_assignment:
                return await self._format_assignment_response(existing_assignment)
            
            # Check if should be included in experiment
            if not self.allocator.should_include_in_experiment(
                experiment.traffic_percentage, participant_hash
            ):
                return None
            
            # Allocate to variant
            active_variants = [
                {
                    'id': str(v.id),
                    'traffic_weight': v.traffic_weight,
                    'configuration': v.configuration,
                    'name': v.name
                }
                for v in experiment.variants if v.is_active
            ]
            
            if not active_variants:
                return None
            
            variant_id = self.allocator.allocate_variant(active_variants, participant_hash)
            
            # Create assignment
            assignment = ParticipantAssignment(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                session_id=session_id,
                participant_hash=participant_hash,
                assignment_method=experiment.allocation_method,
                user_agent=context.get('user_agent') if context else None,
                ip_address=context.get('ip_address') if context else None,
                referrer=context.get('referrer') if context else None,
                assignment_context=context
            )
            
            self.session.add(assignment)
            await self.session.commit()
            await self.session.refresh(assignment)
            
            return await self._format_assignment_response(assignment)
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error assigning participant: {e}")
            return None
    
    async def track_event(
        self, 
        experiment_id: str, 
        participant_hash: str, 
        event_type: str, 
        event_value: Optional[float] = None,
        event_metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> bool:
        """Track an event for experiment analysis."""
        try:
            # Get assignment
            assignment = await self._get_existing_assignment(experiment_id, participant_hash)
            if not assignment:
                return False
            
            # Create event
            event = ExperimentEvent(
                experiment_id=experiment_id,
                variant_id=assignment.variant_id,
                assignment_id=assignment.id,
                event_type=event_type,
                event_value=event_value,
                event_metadata=event_metadata,
                trace_id=trace_id
            )
            
            self.session.add(event)
            await self.session.commit()
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error tracking event: {e}")
            return False
    
    async def analyze_experiment(
        self, 
        experiment_id: str, 
        metric_name: str
    ) -> Optional[Dict[str, Any]]:
        """Perform statistical analysis of experiment results."""
        try:
            # Get experiment data
            experiment_data = await self._get_experiment_data(experiment_id, metric_name)
            if not experiment_data:
                return None
            
            # Determine statistical test
            experiment = experiment_data['experiment']
            control_values = experiment_data['control_values']
            treatment_values = experiment_data['treatment_values']
            
            if experiment.statistical_test == StatisticalTest.CHI_SQUARE:
                # For conversion rate analysis
                control_conversions = len([v for v in control_values if v > 0])
                treatment_conversions = len([v for v in treatment_values if v > 0])
                
                results = self.analyzer.perform_chi_square_test(
                    control_conversions, len(control_values),
                    treatment_conversions, len(treatment_values),
                    experiment.confidence_level
                )
            else:
                # Default to t-test
                results = self.analyzer.perform_t_test(
                    control_values, treatment_values,
                    experiment.confidence_level
                )
            
            if 'error' in results:
                return results
            
            # Store results
            await self._store_experiment_results(
                experiment_id, metric_name, results, experiment.statistical_test
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            return {"error": str(e)}
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        try:
            stmt = select(Experiment).where(Experiment.id == experiment_id)
            result = await self.session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return False
            
            experiment.status = ExperimentStatus.ACTIVE
            experiment.start_date = datetime.utcnow()
            
            if experiment.duration_days:
                experiment.end_date = experiment.start_date + timedelta(days=experiment.duration_days)
            
            await self.session.commit()
            
            # Invalidate cache
            await cache_service.delete_pattern(f"*{experiment.team_id}*", prefix="experiments")
            
            logger.info(f"Started experiment {experiment_id}")
            return True
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error starting experiment: {e}")
            return False
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        try:
            stmt = select(Experiment).where(Experiment.id == experiment_id)
            result = await self.session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return False
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_date = datetime.utcnow()
            
            await self.session.commit()
            
            # Invalidate cache
            await cache_service.delete_pattern(f"*{experiment.team_id}*", prefix="experiments")
            
            logger.info(f"Stopped experiment {experiment_id}")
            return True
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error stopping experiment: {e}")
            return False
    
    # Helper methods
    
    async def _get_existing_assignment(
        self, 
        experiment_id: str, 
        participant_hash: str
    ) -> Optional[ParticipantAssignment]:
        """Get existing participant assignment."""
        stmt = select(ParticipantAssignment).options(
            selectinload(ParticipantAssignment.variant)
        ).where(
            and_(
                ParticipantAssignment.experiment_id == experiment_id,
                ParticipantAssignment.participant_hash == participant_hash
            )
        )
        
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _format_assignment_response(
        self, 
        assignment: ParticipantAssignment
    ) -> Dict[str, Any]:
        """Format assignment for response."""
        return {
            "experiment_id": str(assignment.experiment_id),
            "variant_id": str(assignment.variant_id),
            "variant_name": assignment.variant.name,
            "assigned_at": assignment.assigned_at,
            "configuration": assignment.variant.configuration
        }
    
    async def _get_experiment_data(
        self, 
        experiment_id: str, 
        metric_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get experiment data for analysis."""
        # Get experiment
        stmt = select(Experiment).where(Experiment.id == experiment_id)
        result = await self.session.execute(stmt)
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            return None
        
        # Get events grouped by variant
        events_stmt = select(ExperimentEvent).join(ExperimentVariant).where(
            and_(
                ExperimentEvent.experiment_id == experiment_id,
                ExperimentEvent.event_type == metric_name
            )
        )
        
        events_result = await self.session.execute(events_stmt)
        events = events_result.scalars().all()
        
        # Separate control and treatment values
        control_values = []
        treatment_values = []
        
        for event in events:
            if event.variant.variant_type == VariantType.CONTROL:
                control_values.append(event.event_value or 0)
            else:
                treatment_values.append(event.event_value or 0)
        
        return {
            "experiment": experiment,
            "control_values": control_values,
            "treatment_values": treatment_values
        }
    
    async def _store_experiment_results(
        self, 
        experiment_id: str, 
        metric_name: str, 
        results: Dict[str, Any],
        statistical_test: str
    ) -> None:
        """Store experiment analysis results."""
        result_record = ExperimentResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            control_sample_size=results.get('control_sample_size'),
            treatment_sample_size=results.get('treatment_sample_size'),
            control_mean=results.get('control_mean'),
            treatment_mean=results.get('treatment_mean'),
            control_std=results.get('control_std'),
            treatment_std=results.get('treatment_std'),
            absolute_effect=results.get('absolute_effect'),
            relative_effect=results.get('relative_effect'),
            p_value=results.get('p_value'),
            confidence_interval_lower=results.get('confidence_interval_lower'),
            confidence_interval_upper=results.get('confidence_interval_upper'),
            is_significant=results.get('is_significant'),
            statistical_test_used=statistical_test,
            test_statistic=results.get('test_statistic'),
            degrees_of_freedom=results.get('degrees_of_freedom'),
            pooled_variance=results.get('pooled_variance'),
            effect_size_cohens_d=results.get('effect_size_cohens_d'),
            analysis_metadata=results
        )
        
        self.session.add(result_record)
        await self.session.commit() 