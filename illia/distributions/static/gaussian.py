# Libraries
from typing import Any, Optional

import illia.distributions.static as static


class GaussianDistribution(static.StaticDistribution):
    
    def __init__(
        self,
        mu: float,
        std: float,
        backend: Optional[str] = "torch",
    ) -> None:
        
        # Call super class constructor
        super(GaussianDistribution, self).__init__()

        # Set attributes
        self.backend = backend

        # Choose backend
        if backend == "torch":
            # Import torch part
            from illia.distributions.static.torch import gaussian
        elif backend == "tf":
            # Import tensorflow part
            from illia.distributions.static.tf import gaussian
        else:
            raise ValueError("Invalid backend value")
        
        # Define distribution based on the imported library
        self.distribution = gaussian.GaussianDistribution(mu, std)

    # Overriding method
    def log_prob(self, x: Any) -> Any:
        return self.distribution.log_prob(x)
