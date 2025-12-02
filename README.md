📊 Projeto de Classificação Binária com MLP e Autoajuste
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/TensorFlow-2.0+-orange.svg
https://img.shields.io/badge/License-MIT-green.svg

Um pipeline completo de Machine Learning para classificação binária utilizando redes neurais MLP com ajuste automático de hiperparâmetros, pré-processamento inteligente e técnicas avançadas de balanceamento.

✨ Funcionalidades
🔧 Pré-processamento automático: Tratamento de dados ausentes e codificação de variáveis

⚖️ Balanceamento de classes: Implementação de SMOTE para datasets desbalanceados

🤖 AutoML integrado: Busca automática de melhores hiperparâmetros com KerasTuner

📈 Avaliação completa: Métricas detalhadas, matriz de confusão e curvas ROC

🛡️ Prevenção de overfitting: Dropout, BatchNorm e Early Stopping

🚀 Resultados Principais
Métrica	Classe 0	Classe 1	Global
Acurácia	-	-	93,06%
Precisão	100%	88%	94%
Recall	86%	100%	93%
F1-Score	93%	94%	93%
🏗️ Arquitetura do Modelo
text
MLP com ajuste automático:
├── Camada de entrada (32-128 unidades)
├── Batch Normalization
├── Dropout (10-50%)
├── 1-3 camadas ocultas (32-128 unidades)
└── Camada de saída (softmax)
📦 Instalação
bash
# Clone o repositório
git clone https://github.com/seu-usuario/mlp-classifier.git
cd mlp-classifier

# Crie um ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
📋 Dependências
Python 3.8+

TensorFlow 2.x

scikit-learn

pandas, numpy

matplotlib, seaborn

imbalanced-learn

keras-tuner

🎯 Uso Rápido
python
import pandas as pd
from mlp_pipeline import run_mlp_model

# Carregue seus dados
df = pd.read_csv('seu_dataset.csv')

# Execute o pipeline completo
run_mlp_model(
    df=df,
    target_column='status',  # Substitua pelo nome da sua coluna alvo
    balance_data=True        # Ativa/desativa SMOTE
)
📊 Dataset
O projeto inclui suporte para qualquer dataset estruturado. Exemplo utilizado:

Origem: Dados públicos (URL do Google Sheets)

Alvo: Coluna 'status' (classificação binária)

Pré-processamento: Automático e adaptativo

⚙️ Pipeline Completo
Tratamento de nulos → Mediana (numéricos) / 'não informado' (categóricos)

Label Encoding → Conversão automática de categóricas

Normalização MinMax → Features entre 0 e 1

Balanceamento SMOTE → Opcional

Divisão estratificada → 80% treino, 20% teste

Busca de hiperparâmetros → Random Search (5 trials)

Treinamento com early stopping → Prevenção de overfitting

Avaliação completa → 8 métricas + visualizações

📈 Visualizações Geradas
Matriz de Confusão (valores absolutos e percentuais)

Curva ROC (apenas para classificação binária)

Relatório de Classificação detalhado

🧪 Experimentação
python
# Para experimentar com diferentes configurações:
resultados = run_mlp_model(
    df=df,
    target_column='status',
    balance_data=True,      # Teste com False para datasets balanceados
    # Parâmetros adicionais podem ser adicionados aqui
)
📚 Aprendizados Técnicos
MLP vs modelos tradicionais para classificação

Importância do balanceamento em datasets desiguais

Tuning automático vs manual de hiperparâmetros

Regularização em redes neurais profundas

Interpretabilidade de modelos de deep learning

🤝 Contribuição
Contribuições são bem-vindas! Siga estes passos:

Fork o projeto

Crie uma branch (git checkout -b feature/nova-feature)

Commit suas mudanças (git commit -m 'Adiciona nova feature')

Push para a branch (git push origin feature/nova-feature)

Abra um Pull Request

Autor: Jhonney Lima
