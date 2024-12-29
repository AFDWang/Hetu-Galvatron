from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ConfigFactory:
    """Factory class for getting model configurations"""
    
    @staticmethod
    def get_config_json(model_type: str) -> Dict[str, Any]:
        """Get configuration dictionary for a given model type.
        
        Args:
            model_type: Type of model ("gpt", "llama", or "llama2")
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "gpt":
            from tests.models.configs.gpt import GPTConfig
            return GPTConfig().to_dict()
        elif model_type == "gpt256":
            from tests.models.configs.gpt import GPTConfig256
            return GPTConfig256().to_dict()
        elif model_type == "llama":
            from tests.models.configs.llama import LlamaConfig
            return LlamaConfig().to_dict()
        elif model_type == "llama2":
            from tests.models.configs.llama import Llama2Config
            return Llama2Config().to_dict()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")