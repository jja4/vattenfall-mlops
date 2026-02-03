"""
Automated champion model promotion.

This module compares the staging (challenger) model against the current
production (champion) model and promotes the challenger if it performs better.

Promotion rules:
- Challenger test_mae must be lower than champion test_mae
- Challenger test_r2 must not regress by more than 5%
- If no production model exists, staging is promoted automatically

Usage:
    python -m pipeline.promote
    
Environment Variables:
    WANDB_API_KEY: W&B authentication key
"""
import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple

import wandb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# W&B configuration
WANDB_PROJECT = "vattenfall-imbalance-price"
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
MODEL_REGISTRY_NAME = "imbalance-price-model"

# Promotion thresholds
MAE_IMPROVEMENT_REQUIRED = 0.0  # Challenger MAE must be <= champion MAE
R2_REGRESSION_TOLERANCE = 0.05  # Allow up to 5% R² regression


def get_model_metrics(alias: str) -> Optional[Dict[str, Any]]:
    """
    Get metrics for a model by its alias.
    
    Args:
        alias: Model alias ('staging' or 'production')
        
    Returns:
        Dict with model metadata including metrics, or None if not found
    """
    api = wandb.Api()
    
    try:
        # Construct artifact path
        entity = WANDB_ENTITY or api.default_entity
        artifact_path = f"{entity}/{WANDB_PROJECT}/{MODEL_REGISTRY_NAME}:{alias}"
        
        logger.info(f"Fetching model: {artifact_path}")
        artifact = api.artifact(artifact_path)
        
        return {
            "version": artifact.version,
            "created_at": artifact.created_at,
            "metadata": artifact.metadata,
            "aliases": artifact.aliases,
            # Extract key metrics from metadata
            "test_mae": artifact.metadata.get("test_mae"),
            "test_rmse": artifact.metadata.get("test_rmse"),
            "test_r2": artifact.metadata.get("test_r2"),
            "dataset_hash": artifact.metadata.get("dataset_hash"),
        }
        
    except wandb.errors.CommError as e:
        error_msg = str(e).lower()
        if "does not exist" in error_msg or "could not find" in error_msg or "not found" in error_msg:
            logger.warning(f"No model found with alias '{alias}'")
            return None
        raise


def should_promote(staging: Dict[str, Any], production: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Determine if staging model should be promoted to production.
    
    Args:
        staging: Staging model metadata
        production: Production model metadata (may be None)
        
    Returns:
        Tuple of (should_promote, reason)
    """
    # If no production model, promote automatically
    if production is None:
        return True, "No production model exists - promoting staging automatically"
    
    staging_mae = staging.get("test_mae")
    production_mae = production.get("test_mae")
    staging_r2 = staging.get("test_r2")
    production_r2 = production.get("test_r2")
    
    # Check for missing metrics
    if staging_mae is None or production_mae is None:
        return False, f"Missing MAE metrics (staging={staging_mae}, production={production_mae})"
    
    # Rule 1: MAE must improve (lower is better)
    mae_improvement = production_mae - staging_mae
    if staging_mae > production_mae + MAE_IMPROVEMENT_REQUIRED:
        return False, f"MAE regression: staging={staging_mae:.2f} > production={production_mae:.2f}"
    
    # Rule 2: R² must not regress significantly (higher is better)
    if staging_r2 is not None and production_r2 is not None:
        r2_change = staging_r2 - production_r2
        if r2_change < -R2_REGRESSION_TOLERANCE:
            return False, f"R² regression: staging={staging_r2:.4f} < production={production_r2:.4f} (tolerance={R2_REGRESSION_TOLERANCE})"
    
    # All checks passed
    return True, f"MAE improved by {mae_improvement:.2f} EUR/MWh (staging={staging_mae:.2f}, production={production_mae:.2f})"


def promote_model(staging_version: str) -> bool:
    """
    Promote staging model to production by updating aliases.
    
    This:
    1. Removes 'production' alias from current champion
    2. Adds 'archived' alias to old champion
    3. Adds 'production' alias to staging model
    4. Removes 'staging' alias from promoted model
    
    Args:
        staging_version: Version string of staging model (e.g., 'v3')
        
    Returns:
        True if promotion succeeded
    """
    api = wandb.Api()
    entity = WANDB_ENTITY or api.default_entity
    
    # Get staging artifact
    staging_path = f"{entity}/{WANDB_PROJECT}/{MODEL_REGISTRY_NAME}:{staging_version}"
    staging_artifact = api.artifact(staging_path)
    
    # Get current production artifact (if exists)
    try:
        production_path = f"{entity}/{WANDB_PROJECT}/{MODEL_REGISTRY_NAME}:production"
        production_artifact = api.artifact(production_path)
        
        # Archive old production model
        current_aliases = list(production_artifact.aliases)
        if "production" in current_aliases:
            current_aliases.remove("production")
        if "archived" not in current_aliases:
            current_aliases.append("archived")
        
        production_artifact.aliases = current_aliases
        production_artifact.save()
        logger.info(f"Archived old production model: {production_artifact.version}")
        
    except wandb.errors.CommError:
        logger.info("No existing production model to archive")
    
    # Promote staging to production
    new_aliases = list(staging_artifact.aliases)
    if "staging" in new_aliases:
        new_aliases.remove("staging")
    if "production" not in new_aliases:
        new_aliases.append("production")
    
    staging_artifact.aliases = new_aliases
    staging_artifact.save()
    
    logger.info(f"Promoted {staging_version} to production")
    return True


def run_promotion() -> Dict[str, Any]:
    """
    Run the automated promotion pipeline.
    
    Returns:
        Dict with promotion results
    """
    logger.info("=" * 60)
    logger.info("🏆 Automated Champion Promotion")
    logger.info("=" * 60)
    
    # Get staging model
    logger.info("\n📥 Fetching staging model...")
    staging = get_model_metrics("staging")
    
    if staging is None:
        logger.error("No staging model found - nothing to promote")
        return {"promoted": False, "reason": "No staging model found"}
    
    logger.info(f"Staging model: {staging['version']}")
    logger.info(f"  MAE: {staging['test_mae']:.2f}")
    logger.info(f"  R²: {staging['test_r2']:.4f}")
    logger.info(f"  Dataset: {staging['dataset_hash']}")
    
    # Get production model
    logger.info("\n📥 Fetching production model...")
    production = get_model_metrics("production")
    
    if production:
        logger.info(f"Production model: {production['version']}")
        logger.info(f"  MAE: {production['test_mae']:.2f}")
        logger.info(f"  R²: {production['test_r2']:.4f}")
    else:
        logger.info("No production model found (first deployment)")
    
    # Evaluate promotion rules
    logger.info("\n🔍 Evaluating promotion rules...")
    should_promote_model, reason = should_promote(staging, production)
    
    logger.info(f"Decision: {'PROMOTE' if should_promote_model else 'REJECT'}")
    logger.info(f"Reason: {reason}")
    
    # Execute promotion
    if should_promote_model:
        logger.info("\n🚀 Promoting model...")
        success = promote_model(staging["version"])
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ Promotion Complete")
            logger.info("=" * 60)
            logger.info(f"New production model: {staging['version']}")
            
            return {
                "promoted": True,
                "reason": reason,
                "new_production_version": staging["version"],
                "old_production_version": production["version"] if production else None,
                "staging_mae": staging["test_mae"],
                "production_mae": production["test_mae"] if production else None,
            }
    else:
        logger.info("\n" + "=" * 60)
        logger.info("❌ Promotion Rejected")
        logger.info("=" * 60)
        logger.info(f"Staging model {staging['version']} did not meet promotion criteria")
    
    return {
        "promoted": False,
        "reason": reason,
        "staging_version": staging["version"],
        "staging_mae": staging["test_mae"],
        "production_mae": production["test_mae"] if production else None,
    }


def main():
    """Entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated model promotion")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Check promotion rules without promoting")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.dry_run:
            logger.info("DRY RUN - will not actually promote")
            staging = get_model_metrics("staging")
            production = get_model_metrics("production")
            
            if staging:
                should_promote_model, reason = should_promote(staging, production)
                logger.info(f"Would promote: {should_promote_model}")
                logger.info(f"Reason: {reason}")
            return 0
        
        results = run_promotion()
        
        # Exit with appropriate code
        if results.get("promoted"):
            return 0
        else:
            # Not promoted is not an error, just means challenger wasn't better
            return 0
        
    except Exception as e:
        logger.error(f"Promotion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
