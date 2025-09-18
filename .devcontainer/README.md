# Dev Container Configuration

This directory contains the configuration files required to set up a development
container. These configurations are compatible with GitHub Codespaces, Visual Studio
Code, and JetBrains IDEs, providing a preconfigured environment with all necessary
dependencies for development.

## GitHub Codespaces

To initiate a development container using GitHub Codespaces:

1. Navigate to the repository’s main page.
2. Click the **"Code"** button.
3. Select the **"Codespaces"** tab.
4. Click the **"+"** button to create a new Codespace. The container will initialize
   automatically based on the configuration files in this directory. For more detailed
   instructions, refer to the
   [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository).

## Visual Studio Code

To use the development container in Visual Studio Code:

1. Open the repository’s root folder in Visual Studio Code.
2. When prompted to reopen the folder in a development container, select **"Reopen in
   Container"**. Comprehensive instructions are available in the
   [VS Code Dev Containers Guide](https://code.visualstudio.com/docs/devcontainers/tutorial).

## JetBrains IDEs

To utilize the development container in a JetBrains IDE, such as IntelliJ IDEA or
PyCharm:

1. Open the `.devcontainer/devcontainer.json` file in your chosen IDE.
2. Click the Docker icon that appears in the interface.
3. Follow the guided steps to create and open the development container. More information
   can be found in the
   [JetBrains Dev Container Integration Guide](https://www.jetbrains.com/help/idea/connect-to-devcontainer.html).
