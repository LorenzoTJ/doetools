"""Unit tests for the design_wizard module.

Tests the DesignRecommendation system and suggest_design functionality.
"""

import pytest
from doetools.utils.design_wizard import (
    suggest_design,
    DesignRecommendation,
    FactorAnalyzer,
    _process_levels,
    _next_pow2,
    _min_runs_fractional,
    _cat_levels,
    _cont_levels
)
from doetools.utils.factors import ContinuousFactor


class TestDesignRecommendation:
    
    """Test the DesignRecommendation dataclass."""

    def test_recommendation_creation(self):
        """Test creating a design recommendation."""
        rec = DesignRecommendation(
            design_name="Test Design",
            n_runs=10,
            pros=["Pro 1", "Pro 2"],
            cons=["Con 1"],
            use_case=["Use case 1"],
            additional_info=["Info 1", "Info 2"]
        )
        assert rec.design_name == "Test Design"
        assert rec.n_runs == 10
        assert len(rec.pros) == 2
        assert len(rec.cons) == 1
        assert len(rec.use_case) == 1
        assert len(rec.additional_info) == 2

    def test_recommendation_str_formatting(self):
        """Test string representation of recommendation."""
        rec = DesignRecommendation(
            design_name="Test Design",
            n_runs=10,
            pros=["Efficient"],
            cons=["Limited"],
            use_case=["Screening"],
            additional_info=["Additional info"]
        )
        output = rec.__str__()
        assert "Test Design" in output
        assert "10" in output
        assert "Efficient" in output
        assert "Additional info" in output

class TestFactorAnalyzer:
    
    """Test the FactorAnalyzer class."""

    def test_continuous_factors_only(self, sample_continuous_factors):
        """Test analyzer with only continuous factors."""
        fa = FactorAnalyzer(sample_continuous_factors)
        assert fa.n_continuous == 3
        assert fa.n_categorical == 0
        assert fa.n_process == 3
        assert fa.n_mixture == 0
        assert fa.has_process is True
        assert fa.has_mixture is False
        assert fa.has_categorical is False
        assert fa.is_mixed_problem() is False
        assert fa.has_upper_bounds() is False
    
    def test_categorical_factors_only(self, sample_categorical_factors):
        """Test analyzer with only categorical factors."""
        fa = FactorAnalyzer(sample_categorical_factors)
        assert fa.n_continuous == 0
        assert fa.n_categorical == 3
        assert fa.n_process == 3
        assert fa.n_mixture == 0
        assert fa.has_process is True
        assert fa.has_mixture is False
        assert fa.has_categorical is True
        assert fa.is_mixed_problem() is False
        assert fa.has_upper_bounds() is False

    def test_mixture_factors_only(self, sample_mixture_factors):
        """Test analyzer with only mixture factors."""
        fa = FactorAnalyzer(sample_mixture_factors)
        assert fa.n_continuous == 0
        assert fa.n_categorical == 0
        assert fa.n_process == 0
        assert fa.n_mixture == 4
        assert fa.has_process is False
        assert fa.has_mixture is True
        assert fa.has_categorical is False
        assert fa.is_mixed_problem() is False
        assert fa.has_upper_bounds() is False

    def test_mixed_problem(self, mixed_factors):
        """Test analyzer with both process and mixture factors."""
        fa = FactorAnalyzer(mixed_factors)
        assert fa.n_continuous == 2
        assert fa.n_categorical == 1
        assert fa.n_process == 3
        assert fa.n_mixture == 2
        assert fa.has_process is True
        assert fa.has_mixture is True
        assert fa.has_categorical is True
        assert fa.is_mixed_problem() is True
        assert fa.has_upper_bounds() is False

    def test_constrained_mixture_detection(self, constrained_mixture_factors):
        """Test detection of upper bound constraints in mixture factors."""
        fa = FactorAnalyzer(constrained_mixture_factors)
        assert fa.has_upper_bounds() is True

    def test_validate_empty_factors(self):
        """Test validation with empty factor dictionary."""
        fa = FactorAnalyzer({})
        with pytest.raises(ValueError, match="No factors provided"):
            fa.validate()

class TestHelperFunctions:
    """Test helper functions in design_wizard module."""

    def test_next_pow2(self):
        """Test _next_pow2 function."""
        assert _next_pow2(1) == 1
        assert _next_pow2(2) == 2
        assert _next_pow2(3) == 4
        assert _next_pow2(5) == 8
        assert _next_pow2(9) == 16
        assert _next_pow2(16) == 16
        assert _next_pow2(17) == 32

    def test_min_runs_fractional_resolution_3(self):
        """Test minimum runs for Resolution III fractional factorial."""
        assert _min_runs_fractional(3, 3) == 4   # k+1 = 4
        assert _min_runs_fractional(7, 3) == 8   # k+1 = 8
        assert _min_runs_fractional(10, 3) == 16 # k+1 = 11 -> 16

    def test_min_runs_fractional_resolution_4(self):
        """Test minimum runs for Resolution IV fractional factorial."""
        assert _min_runs_fractional(3, 4) == 8   # 2k = 6 -> 8
        assert _min_runs_fractional(5, 4) == 16  # 2k = 10 -> 16
        assert _min_runs_fractional(8, 4) == 16  # 2k = 16 -> 16

    def test_cont_levels(self, sample_continuous_factors):
        """Test _cont_levels helper function."""
        fa = FactorAnalyzer(sample_continuous_factors)
        levels = _cont_levels(fa)
        assert len(levels) == 3
        assert all(lv == 3 for lv in levels)
    
    def test_cat_levels(self, sample_categorical_factors):
        """Test _cat_levels helper function."""
        fa = FactorAnalyzer(sample_categorical_factors)
        levels = _cat_levels(fa)
        assert len(levels) == 3
        assert levels == [3, 2, 4]
    
    def test_process_levels(self, process_factors):
        """Test _process_levels helper function."""
        fa = FactorAnalyzer(process_factors)
        levels = _process_levels(fa)
        assert len(levels) == 4
        assert levels == [4, 4, 4, 3]

    def test_min_runs_fractional_invalid_resolution(self):
        """Test error handling for invalid resolution."""
        with pytest.raises(ValueError, match="resolution must be 3 or 4"):
            _min_runs_fractional(5, 5)



class TestSuggestDesign:
    
    """Test the main suggest_design function."""
    
    def test_max_experiments_negative(self, sample_continuous_factors):
        """Test error handling for negative max_experiments."""
        with pytest.raises(ValueError, match="max_experiments must be positive"):
            suggest_design(
                factors=sample_continuous_factors,
                phase='screening',
                model_order='linear',
                max_experiments=-1
            )
    
    # --------------------------- PROCESS FACTORS SCREENING TESTS ---------------------------
    
    # 1) Constrained Process Factors - Screening
    # D-Optimal should be suggested
    def test_screening_process_d_optimal(self, sample_continuous_factors):
        
        """Test screening design suggestion for process factors."""
        # Modify factors to have 2 levels for screening
        factors_2level = {
            k: ContinuousFactor(n_levels=2, lower_bound=v.lower_bound, upper_bound=v.upper_bound)
            for k, v in sample_continuous_factors.items()
        }
        
        rec = suggest_design(
            factors=factors_2level,
            phase='screening',
            model_order='2FI',
            max_experiments=None,
            non_rectangular_constraints=True
        )
        
        assert len(rec) > 0
        assert isinstance(rec[0], DesignRecommendation)
        assert any("D-Optimal" in r.design_name for r in rec) 
    
    # 2) Unconstrained Process Factors - Screening
    
    # 2.1) max_experiments large enough for full factorial
    
    # 2.1.1) 2-level factors  
    def test_screening_process_full_factorial_2level(self, sample_continuous_factors):
        """Test screening full factorial suggestion for 2-level process factors."""
        # Modify factors to have 2 levels for screening
        factors_2level = {
            k: ContinuousFactor(n_levels=2, lower_bound=v.lower_bound, upper_bound=v.upper_bound)
            for k, v in sample_continuous_factors.items()
        }
        
        recommendations = suggest_design(
            factors=factors_2level,
            phase='screening',
            model_order='linear',
            max_experiments=20
        )
        
        assert len(recommendations) > 0
        assert isinstance(recommendations[0], DesignRecommendation)
        # Should suggest full factorial for 3 factors at 2 levels (2^3=8 runs)
        design_names = [rec.design_name for rec in recommendations]
        assert any('Full Factorial' in name for name in design_names)
        
    # 2.1.2) Mixed-level factors
    def test_screening_process_full_factorial_mixedlevel(self, screening_process_factors):
        """Test screening full factorial suggestion for mixed-level process factors."""
        # Add a categorical factor to create mixed levels
        factors_mixed = screening_process_factors.copy()
                
        recommendations = suggest_design(
            factors=factors_mixed,
            phase='screening',
            model_order='linear',
            max_experiments=80
        )
        
        assert len(recommendations) > 0
        # Should suggest full factorial for mixed levels (2*2*2*3=24 runs)
        design_names = [rec.design_name for rec in recommendations]
        assert any('Full Factorial' in name for name in design_names)
    
    # 2.2) max_experiments not large enough for full factorial
    
    # 2.2.1) 3-levels categorical factors -> Fail + D-Optimal
    def test_screening_process_3level_cat_failure(self, screening_process_factors):
        """Test screening suggestion failure for 3-level categorical process factors."""
        # Add a 3-level categorical factor to exceed max_experiments
        factors_3level_cat = screening_process_factors.copy()
                        
        recommendations = suggest_design(
            factors=factors_3level_cat,
            phase='screening',
            model_order='linear',
            max_experiments=20
        )
        
        assert len(recommendations) > 0
        # Should suggest D-Optimal due to inability to run full factorial
        design_names = [rec.design_name for rec in recommendations]
        assert any('No Suitable Design' in name for name in design_names)
        
    # 2.2.2) 2-levels factors / No interactions -> Plackett-Burman
    def test_screening_process_plackett_burman(self, sample_pb_factors):
        """Test screening Plackett-Burman suggestion for 2-level process factors."""
        
        recommendations = suggest_design(
            factors=sample_pb_factors,
            phase='screening',
            model_order='linear',
            max_experiments=20
        )
        
        assert len(recommendations) > 0
        # Should suggest Plackett-Burman for 5 factors at 2 levels (8 runs)
        design_names = [rec.design_name for rec in recommendations]
        assert any('Plackett' in name and 'Burman' in name for name in design_names)
    
    # 2.2.3) 2-levels factors / interactions -> FrFIV
    def test_screening_process_fractional_factorial_IV(self, sample_pb_factors):
        """Test screening fractional factorial (Resolution IV) suggestion."""
        
        recommendations = suggest_design(
            factors=sample_pb_factors,
            phase='screening',
            model_order='2FI',
            max_experiments=20
        )
        
        assert len(recommendations) > 0
        # Should suggest Fractional Factorial Resolution IV for 5 factors at 2 levels (16 runs)
        design_names = [rec.design_name for rec in recommendations]
        assert any('Fractional Factorial Design (Resolution IV)' in name for name in design_names)

    # 2.2.4) 2-levels factors / interactions -> FrFIII
    def test_screening_process_fractional_factorial_III(self, sample_pb_factors):
            """Test screening fractional factorial (Resolution III) suggestion."""
            
            recommendations = suggest_design(
                factors=sample_pb_factors,
                phase='screening',
                model_order='2FI',
                max_experiments=10
            )
            
            assert len(recommendations) > 0
            # Should suggest Fractional Factorial Resolution III for 5 factors at 2 levels (8 runs)
            design_names = [rec.design_name for rec in recommendations]
            assert any('Fractional Factorial Design (Resolution III)' in name for name in design_names)
   
    # 2.2.5) Not enough runs for FrFIII -> Insufficient Budget
    def test_screening_process_insufficient_budget(self, sample_pb_factors):
            """Test screening suggestion failure due to insufficient budget."""
            
            recommendations = suggest_design(
                factors=sample_pb_factors,
                phase='screening',
                model_order='2FI',
                max_experiments=6
            )
            
            assert len(recommendations) > 0
            # Should indicate insufficient budget for any design
            design_names = [rec.design_name for rec in recommendations]
            assert any('Insufficient Budget' in name for name in design_names)
            
    # -------------------------- MIXED FACTORS SCREENING TESTS --------------------------
    
    # Mixed process-mixture factors -> Separate Screening Strategy
    def test_mixed_problem_screening(self, mixed_factors): 
        
        """Test screening suggestion for mixed process-mixture factors."""
        recommendations = suggest_design(
            factors=mixed_factors,
            phase='screening',
            model_order='linear',
            max_experiments=50
        )
        
        assert len(recommendations) > 0
        # Should suggest separate screening strategy
        design_names = [rec.design_name for rec in recommendations]
        assert any('Separate Screening Strategy'in name for name in design_names)

    # --------------------------- MIXTURE FACTORS SCREENING TESTS ---------------------------
    
    # 1) Mixture Factors - Screening
    
    # 1.1) Constrained Mixture Factors
    def test_mixture_screening_upper_bound(self, constrained_mixture_factors):
        """Test screening design suggestion for constrained mixture factors."""
        recommendations = suggest_design(
            factors=constrained_mixture_factors,
            phase='screening',
            model_order='linear',
            max_experiments=30
        )
        
        assert len(recommendations) > 0
        # Should suggest D-Optimal for constrained mixture screening
        design_names = [rec.design_name for rec in recommendations]
        assert any('D-Optimal' in name for name in design_names)
    
    def test_mixture_screening_constrained(self, sample_mixture_factors):
        """Test screening design suggestion for constrained mixture factors."""
        recommendations = suggest_design(
            factors=sample_mixture_factors,
            phase='screening',
            model_order='linear',
            max_experiments=30,
            non_rectangular_constraints=True
        )
        
        assert len(recommendations) > 0
        # Should suggest D-Optimal for constrained mixture screening
        design_names = [rec.design_name for rec in recommendations]
        assert any('D-Optimal' in name for name in design_names)
    
    # 1.2) Unconstrained Mixture Factors
    
    # 1.2.1) Simplex-Centroid -> Enough runs
    def test_mixture_screening_simplex_centroid(self, sample_mixture_factors):
        """Test screening Simplex-Centroid suggestion for unconstrained mixture factors."""
        recommendations = suggest_design(
            factors=sample_mixture_factors,
            phase='screening',
            model_order='linear',
            max_experiments=20
        )
        
        assert len(recommendations) > 0
        # Should suggest Simplex-Centroid for 3 mixture factors (7 runs)
        design_names = [rec.design_name for rec in recommendations]
        assert any('Simplex-Centroid' in name for name in design_names)
        
    # 1.2.2) D-Optimal -> Not enough runs
    def test_mixture_screening_d_optimal(self, sample_mixture_factors):
        """Test screening D-Optimal suggestion for unconstrained mixture factors."""
        recommendations = suggest_design(
            factors=sample_mixture_factors,
            phase='screening',
            model_order='linear',
            max_experiments=5
        )
        
        assert len(recommendations) > 0
        # Should suggest D-Optimal due to insufficient runs for Simplex-Centroid
        design_names = [rec.design_name for rec in recommendations]
        assert any('Insufficient Budget' in name for name in design_names)
    
    # --------------------------- MIXED FACTORS OPTIMIZATION TESTS ---------------------------
    
    # 1) Mixed process-mixture factors - Optimization
    
    # 1.1) performed_exp=False -> D-Optimal
    def test_mixed_problem_optimization_no_performed(self, mixed_factors):
        """Test optimization suggestion for mixed process-mixture factors without performed experiments."""
        recommendations = suggest_design(
            factors=mixed_factors,
            phase='optimization',
            model_order='quadratic',
            max_experiments=50,
            performed_exp=False
        )
        
        assert len(recommendations) > 0
        # Should suggest D-Optimal for mixed process-mixture optimization
        design_names = [rec.design_name for rec in recommendations]
        assert any('D-Optimal Design' in name for name in design_names)
    
    # 1.2) performed_exp=True -> Augmentation
    def test_mixed_problem_optimization_with_performed(self, mixed_factors):
        """Test optimization suggestion for mixed process-mixture factors with performed experiments."""
        recommendations = suggest_design(
            factors=mixed_factors,
            phase='optimization',
            model_order='quadratic',
            max_experiments=50,
            performed_exp=True
        )
        
        assert len(recommendations) > 0
        # Should suggest Augmentation for mixed process-mixture optimization
        design_names = [rec.design_name for rec in recommendations]
        assert any('Augmentation' in name for name in design_names)
    
    # --------------------------- PROCESS FACTORS OPTIMIZATION TESTS ---------------------------
    
    # 1) Process Factors - Optimization
    
    # 1.1) Unconstrained Process Factors
    
    # 1.1.1) Categorical Factors
    
    # 1.1.1.1) performed_exp = True -> Augmentation
    def test_optimization_process_categorical_augmentation(self, process_factors):
        """Test optimization Augmentation suggestion for categorical process factors."""
        # Add a categorical factor to create mixed levels
        factors_mixed = process_factors.copy()
                
        recommendations = suggest_design(
            factors=factors_mixed,
            phase='optimization',
            model_order='quadratic',
            performed_exp=True
        )
        
        assert len(recommendations) > 0
        # Should suggest Augmentation when performed_exp=True
        design_names = [rec.design_name for rec in recommendations]
        assert any('Augmentation' in name for name in design_names)
        
    # 1.1.1.2) performed_exp = False -> FF / D-Optimal
    def test_optimization_process_categorical_no_augmentation(self, process_factors):
        """Test optimization suggestion for categorical process factors without augmentation."""
        # Add a categorical factor to create mixed levels
        factors_mixed = process_factors.copy()
                
        recommendations = suggest_design(
            factors=factors_mixed,
            phase='optimization',
            model_order='quadratic',
            performed_exp=False
        )
        
        assert len(recommendations) > 0
        # Should suggest FF or D-Optimal when performed_exp=False
        design_names = [rec.design_name for rec in recommendations]
        assert any('Full Factorial' in name or 'D-Optimal' in name for name in design_names)
    
    # 1.1.2) Continuous Factors
    def test_optimization_process_continuous(self, sample_continuous_factors):
        """Test optimization design suggestion for continuous process factors."""
        recommendations = suggest_design(
            factors=sample_continuous_factors,
            phase='optimization',
            model_order='quadratic'
        )
        
        assert len(recommendations) > 0
        # Should suggest CCD or BBD for 3 continuous factors
        design_names = [rec.design_name for rec in recommendations]
        assert any('Central Composite' in name or 'Box-Behnken' in name 
                  for name in design_names)

    # 1.2) Constrained Process Factors
    
    # 1.2.1) performed_exp=False -> D-Optimal
    def test_optimization_process_d_optimal(self, sample_continuous_factors):
        """Test optimization D-Optimal suggestion for constrained process factors."""
        recommendations = suggest_design(
            factors=sample_continuous_factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=False,
            non_rectangular_constraints=True
        )
        
        assert len(recommendations) > 0
        # Should suggest D-Optimal when constraints are present
        design_names = [rec.design_name for rec in recommendations]
        assert any('D-Optimal Design' in name for name in design_names)
        
    # 1.2.2) performed_exp=True -> Augmentation
    def test_optimization_process_augmentation(self, sample_continuous_factors):
        """Test optimization Augmentation suggestion for constrained process factors."""
        recommendations = suggest_design(
            factors=sample_continuous_factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=True,
            non_rectangular_constraints=True
        )  
        assert len(recommendations) > 0
        # Should suggest Augmentation when constraints are present
        design_names = [rec.design_name for rec in recommendations]
        assert any('Augmentation' in name for name in design_names)
    
    # --------------------------- MIXTURE FACTORS OPTIMIZATION TESTS ---------------------------
    
    # 1) Mixture Factors - Optimization
    
    # 1.1) Unconstrained Mixture Factors
    
    # 1.1.1) performed_exp = True -> Augmentation
    def test_mixture_optimization_augmentation(self, sample_mixture_factors):
        """Test mixture optimization Augmentation suggestion."""
        recommendations = suggest_design(
            factors=sample_mixture_factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=True
        )
        
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('Augmentation' in name for name in design_names)
    
    # 1.1.2) performed_exp = False -> Simplex-Lattice / D-Optimal
    def test_mixture_optimization_no_augmentation(self, sample_mixture_factors):
        """Test mixture optimization suggestion without augmentation."""
        recommendations = suggest_design(
            factors=sample_mixture_factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=False
        )
        
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('Simplex-Lattice' in name or 'D-Optimal' in name for name in design_names)
    
    # 1.2) Constrained Mixture Factors
    
    # 1.2.1) performed_exp=False -> D-Optimal
    def test_mixture_optimization_d_optimal(self, constrained_mixture_factors):
        """Test constrained mixture optimization D-Optimal suggestion."""
        recommendations = suggest_design(
            factors=constrained_mixture_factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=False
        )
        
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('D-Optimal Design' in name for name in design_names)
    
    # 1.2.2) performed_exp=True -> Augmentation
    def test_mixture_optimization_constrained_augmentation(self, constrained_mixture_factors):
        """Test constrained mixture optimization Augmentation suggestion."""
        recommendations = suggest_design(
            factors=constrained_mixture_factors,
            phase='optimization',
            model_order='quadratic',
            performed_exp=True
        )
        
        assert len(recommendations) > 0
        design_names = [rec.design_name for rec in recommendations]
        assert any('Augmentation' in name for name in design_names)

    def test_invalid_phase(self, sample_continuous_factors):
        """Test error handling for invalid phase."""
        with pytest.raises(ValueError, match="Invalid phase"):
            suggest_design(
                factors=sample_continuous_factors,
                phase='invalid_phase'
            )

    def test_invalid_max_experiments(self, sample_continuous_factors):
        """Test error handling for invalid max_experiments."""
        with pytest.raises(ValueError, match="max_experiments must be positive"):
            suggest_design(
                factors=sample_continuous_factors,
                phase='screening',
                max_experiments=0
            )

    @pytest.mark.parametrize("model_order", ["linear", "2FI", "quadratic"])
    def test_different_model_orders(self, sample_continuous_factors, model_order):
        """Test suggestions with different model orders."""
        factors_2level = {
            k: ContinuousFactor(n_levels=2, lower_bound=v.lower_bound, upper_bound=v.upper_bound)
            for k, v in sample_continuous_factors.items()
        }
        
        recommendations = suggest_design(
            factors=factors_2level,
            phase='screening',
            model_order=model_order
        )
        
        assert len(recommendations) > 0
