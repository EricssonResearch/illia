# Dev Container Configuration

This directory contains the configuration files for setting up a development container.
These configurations are compatible with **GitHub Codespaces**, **Visual Studio Code**,
and **JetBrains IDEs**, and provide a pre-configured environment with all necessary
dependencies for development.

## GitHub Codespaces

To launch a dev container using GitHub Codespaces:

1. Navigate to the repository's main page.
2. Click the **"Code"** button.
3. Select the **"Codespaces"** tab.
4. Click the **"+"** button to create a new codespace.

The container will be initialized automatically using the configurations in this
directory.

[GitHub Codespaces Documentation](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository)

## Visual Studio Code

To use the dev container in VS Code:

1. Open the root folder of the repository in Visual Studio Code.
2. A prompt will appear asking if you want to reopen the folder in a dev container.
3. Confirm by selecting **"Reopen in Container"**.

[VS Code Dev Containers Guide](https://code.visualstudio.com/docs/devcontainers/tutorial)

## JetBrains IDEs

To open the dev container in a JetBrains IDE (e.g., IntelliJ IDEA, PyCharm):

1. Open the `.devcontainer/devcontainer.json` file in your IDE.
2. Click the Docker icon that appears in the UI.
3. Follow the prompts to create and open the dev container.

[JetBrains Dev Container Integration Guide](https://www.jetbrains.com/help/idea/connect-to-devcontainer.html)
