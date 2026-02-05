# LangChain Project with Environment Variables

Este projeto demonstra como usar variáveis de ambiente de forma segura em aplicações LangChain usando python-dotenv.

## 🚀 Configuração Inicial

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente

1. **Copie o arquivo .env** (já criado no projeto)
2. **Edite o arquivo .env** e substitua os valores placeholder pelas suas chaves reais:

```env
# OpenAI API Key (required for OpenAI models)
OPENAI_API_KEY=sua-chave-openai-aqui

# Google API Keys (for Google services)
GOOGLE_API_KEY=sua-chave-google-aqui
GOOGLE_CSE_ID=seu-google-cse-id-aqui

# Hugging Face API Key (for Hugging Face models)
HUGGINGFACE_API_KEY=sua-chave-huggingface-aqui

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=sua-chave-anthropic-aqui```

### 3. Executar o Projeto

```bash
# Executar o arquivo principal
python src/main.py

# Executar exemplos de uso do LangChain
python src/example_langchain_usage.py
```

## 📁 Estrutura do Projeto

```
├── .env                           # Variáveis de ambiente (NÃO commitado)
├── .gitignore                     # Já configurado para ignorar .env
├── requirements.txt               # Dependências (inclui python-dotenv)
├── src/
│   ├── main.py                   # Arquivo principal com demonstração
│   └── example_langchain_usage.py # Exemplos de uso do LangChain
└── README.md                     # Este arquivo
```

## 🔧 Como Funciona

### 1. Carregamento das Variáveis

```python
from dotenv import load_dotenv
import os

# Carrega variáveis do arquivo .env
load_dotenv()

# Acessa as variáveis
api_key = os.getenv("OPENAI_API_KEY")
```

### 2. Uso com LangChain

```python
from langchain_openai import ChatOpenAI

# A chave é carregada automaticamente do .env
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)
```

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