# LangChain Project with Environment Variables

Este projeto demonstra como usar variáveis de ambiente de forma segura em aplicações LangChain usando python-dotenv.

## 📱 Quick Start for Mobile Devs (Android/Kotlin)

If you're coming from an Android background, here's how to think about this project:

1.  **Dependency Injection:** We use a custom DI Container in `src/core/dependency_injection.py`. It's like a simplified version of **Koin** or **Hilt**.
2.  **Activity/Orchestration:** `src/main.py` is your `Launcher`. It hands off control to `InteractiveCLI` in `src/core/cli_service.py`, which acts as your `MainActivity` logic.
3.  **Data Classes:** We use Python `@dataclass` for model configurations (`ModelConfig`). This is equivalent to Kotlin `data class`.
4.  **Interfaces:** Check `src/core/interfaces.py`. We use `Protocols` and `ABCs`, which are the Python equivalent of Kotlin `interface` and `abstract class`.
5.  **Environment:** The `.env` file is like your `local.properties` or `BuildConfig` - keep your API keys there!

For a deeper dive, see our [Developer Guide for Android Devs](docs/DEVELOPER_GUIDE.md).

## 🚀 Configuração Inicial

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente

1. **Copie o arquivo .env** (já criado no projeto)
2. **Edite o arquivo .env** e substitua os valores placeholder pelas suas chaves reais:

```env
# Google API Keys (for Google services)
GOOGLE_API_KEY=sua-chave-google-aqui
GOOGLE_CSE_ID=seu-google-cse-id-aqui

# Hugging Face API Key (for Hugging Face models)
HUGGINGFACE_API_KEY=sua-chave-huggingface-aqui

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=sua-chave-anthropic-aqui
```

### 3. Executar o Projeto

```bash
# Executar o arquivo principal
python src/main.py

# Executar exemplos de uso do LangChain
python src/tests/test_claude.py
```

## 📁 Estrutura do Projeto

```
├── data/                          # Persistent data (token usage, JSON exports)
├── docs/                          # Project documentation and guides
├── logs/                          # Application logs
├── src/                           # Source code
│   ├── main.py                    # Main App Launcher (Bootstrapper)
│   └── core/                      # Core business logic (DI, Models, Services)
│       ├── cli_service.py         # Main Orchestrator (MainActivity logic)
│       └── dependency_injection.py # DI Container (Hilt/Koin equivalent)
├── tests/                         # Unit tests and usage examples
├── .env                           # Environment variables (NOT committed)
├── .gitignore                     # Configured to ignore .env and logs
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Orchestration Logic

The project follows a clean separation of concerns:

1.  **Entry Point (`src/main.py`)**: Minimal code to initialize the `DIContainer` and start the CLI.
2.  **DI Container (`src/core/dependency_injection.py`)**: Registers all services (Singletons/Factories) and manages their lifecycles.
3.  **CLI Service (`src/core/cli_service.py`)**: The "Brain" of the UI. It handles the menu loop and calls the appropriate services based on user input.
4.  **Model Factory (`src/core/models/model_factory.py`)**: Dynamically creates LLM instances (Claude, MiniMax, etc.) with all their dependencies injected.
5.  **Interfaces (`src/core/interfaces.py`)**: Defines strict contracts that all models and services must follow, ensuring the code remains modular and testable.

## 🔒 Segurança

- ✅ O arquivo `.env` está no `.gitignore` - nunca será commitado
- ✅ Use diferentes arquivos `.env` para diferentes ambientes
- ✅ Nunca coloque chaves de API diretamente no código
- ✅ Use valores padrão seguros quando as variáveis não existirem

## 🌍 Ambientes

O projeto suporta diferentes ambientes através da variável `ENVIRONMENT`:

- `development` - Modo de desenvolvimento com debug ativo
- `production` - Modo de produção otimizado

## 📚 Recursos Adicionais

- [Documentação do python-dotenv](https://python-dotenv.readthedocs.io/)
- [Documentação do LangChain](https://python.langchain.com/)
- [Boas práticas de segurança para API keys](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

## 🆘 Solução de Problemas

### Erro: "API key not configured"
- Verifique se o arquivo `.env` existe na raiz do projeto
- Confirme se as chaves estão definidas corretamente no `.env`
- Execute `python src/main.py` para verificar o status das chaves

### Erro: "Module not found"
- Execute `pip install -r requirements.txt`
- Verifique se está no ambiente virtual correto