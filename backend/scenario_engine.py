from typing import Dict, Any


class ScenarioEngine:
    """
    Applies restoration interventions to baseline forest-loss predictions.
    """

    def __init__(self):
        pass

    def simulate(
            self,
            baseline: Dict[str, Any],
            scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate the impact of restoration interventions.
        """

        yearly_predictions = baseline.get("yearly_predictions", [])

        modified_predictions = []

        for prediction in yearly_predictions:
            original_loss = prediction["predicted_loss"]

            reduction_factor = (
                    scenario["agricultureReduction"] * 0.003 +
                    scenario["restorationInvestment"] * 0.002 +
                    scenario["protectedAreas"] * 0.002 +
                    scenario["enforcement"] * 0.002 +
                    scenario["indigenousProtection"] * 0.001
            )

            reduction_factor = min(reduction_factor, 0.8)

            simulated_loss = original_loss * (1 - reduction_factor)

            modified_predictions.append({
                "year": prediction["year"],
                "baseline_loss": original_loss,
                "simulated_loss": simulated_loss
            })

        total_simulated_loss = sum(
            p["simulated_loss"]
            for p in modified_predictions
        )

        return {
            "projected_loss": total_simulated_loss,
            "yearly_predictions": modified_predictions,
            "intervention_effectiveness": reduction_factor
        }