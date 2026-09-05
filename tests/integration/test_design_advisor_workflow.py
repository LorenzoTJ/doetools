"""Integration tests for design_advisor workflow.

Tests the complete workflow from factor definition through design suggestion.
"""

import pytest
from doetools.utils.design_advisor import suggest_design
from doetools.utils.factors import ContinuousFactor, CategoricalFactor, MixtureFactor


@pytest.mark.integration
class TestDesignAdvisorWorkflow:
    """Integration tests for complete design-advisor workflows."""

    def test_full_screening_workflow(self):
        """Test a complete screening workflow from factor definition to recommendation."""
        # Define factors
        factors = {
            'Temperature': ContinuousFactor(n_levels=2, lower_bound=20, upper_bound=80),
            'Pressure': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=5),
            'pH': ContinuousFactor(n_levels=2, lower_bound=3, upper_bound=9),
            'Time': ContinuousFactor(n_levels=2, lower_bound=1, upper_bound=10),
            'Concentration': ContinuousFactor(n_levels=2, lower_bound=0.1, upper_bound=1.0)
        }
        
        # Get screening recommendations
        recommendations = suggest_design(
            factors=factors,
            phase='screening',
            model_order='linear',
            max_experiments=20
        )
        
        # Verify recommendations
        assert len(recommendations) > 0
        assert recommendations[0].design_name is not None
        assert recommendations[0].n_runs is not None

    def test_full_optimization_workflow(self):
        """Test a complete optimization workflow."""
        # Define factors for optimization
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5),
            'Time': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=10)
        }
        
        # Get optimization recommendations
        recommendations = suggest_design(
            factors=factors,
            phase='optimization',
            model_order='quadratic'
        )
        
        # Should recommend RSM designs
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('Central Composite' in name or 'Box-Behnken' in name 
                  for name in design_names)

    def test_sequential_experimentation_workflow(self):
        """Test sequential experimentation (augmentation) workflow."""
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5)
        }
        
        # Initial design
        initial_recs = suggest_design(
            factors=factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=False
        )
        
        # Augmentation with constraints (where augmentation is supported)
        augment_recs = suggest_design(
            factors=factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=True,
            non_rectangular_constraints=True
        )
        
        # Verify both return recommendations
        assert len(initial_recs) > 0
        assert len(augment_recs) > 0
        
        # Augmentation should mention augmentation
        augment_names = [rec.design_name for rec in augment_recs]
        assert any('Augmentation' in name for name in augment_names)

    def test_mixture_optimization_workflow(self):
        """Test mixture design optimization workflow."""
        # Define mixture components
        mixture_factors = {
            'Solvent_A': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'Solvent_B': MixtureFactor(lower_bound=0.0, upper_bound=1.0),
            'Solvent_C': MixtureFactor(lower_bound=0.0, upper_bound=1.0)
        }
        
        # Get mixture design recommendations
        recommendations = suggest_design(
            factors=mixture_factors,
            phase='optimization',
            model_order='quadratic'
        )
        
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('Simplex' in name for name in design_names)

    def test_constrained_optimization_workflow(self):
        """Test optimization with constraints."""
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80),
            'Pressure': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=5),
            'Time': ContinuousFactor(n_levels=3, lower_bound=1, upper_bound=10)
        }
        
        # With non-rectangular constraints
        recommendations = suggest_design(
            factors=factors,
            phase='optimization',
            model_order='quadratic',
            non_rectangular_constraints=True
        )
        
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('D-Optimal' in name for name in design_names)

    def test_categorical_factors_workflow(self):
        """Test workflow with categorical factors."""
        factors = {
            'Temperature': ContinuousFactor(n_levels=3, lower_bound=20, upper_bound=80),
            'Catalyst': CategoricalFactor(levels=['A', 'B', 'C']),
            'Solvent': CategoricalFactor(levels=['Water', 'Ethanol'])
        }
        
        recommendations = suggest_design(
            factors=factors,
            phase='optimization',
            model_order='quadratic'
        )
        
        assert len(recommendations) > 0

    @pytest.mark.slow
    def test_large_factor_space_workflow(self):
        """Test workflow with many factors (slow test)."""
        # Create many factors
        factors = {
            f'Factor_{i}': ContinuousFactor(n_levels=2, lower_bound=0, upper_bound=10)
            for i in range(10)
        }
        
        recommendations = suggest_design(
            factors=factors,
            phase='screening',
            model_order='linear',
            max_experiments=50
        )
        
        assert len(recommendations) > 0
