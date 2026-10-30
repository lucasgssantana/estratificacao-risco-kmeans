## Paper: teste da hipótese de contínuo

`src/pipeline_paper.py` estende o pipeline do TCC com os testes formais que o
manuscrito "Gradiente ou categorias?" reporta. Reutiliza a escoragem de
`pipeline_final.py`; a Etapa 0 reproduz a solução de referência (K = 2: 135/103,
silhueta 0,212) e interrompe a execução se ela não for recuperada.

Etapas: nulos da silhueta (normal multivariada e cópula gaussiana, B = 1000);
GMM K = 1–5 em quatro covariâncias com BIC/aBIC; BLRT paramétrico e de cópula
(`blrt_parametrico.py`, B = 1000); bootstrap, split-half não circular e 100
sementes para K = 3; quatro sensibilidades à composição das dimensões; comparação
perfil vs. DERS dicotomizada vs. DERS contínua na predição da DASS-21.

Execução: `DATA_DIR=data OUT_DIR=outputs/paper python src/pipeline_paper.py`.
Aceita `BASE_ANONIMIZADA_n238.csv` (depósito, itens brutos) ou as bases internas.
Sementes 42 e 2026. Tempo aproximado: 15–40 min, dominado pelo BLRT.

Resultados de referência (n = 238): silhueta K = 2 não excede o nulo de cópula
(p = 0,057); BIC seleciona K = 1 em covariância compartilhada e plena; BLRT sob
cópula não rejeita K = 1; conclusão mantida nas quatro sensibilidades (ARI 0,65–0,97).

Ambiente verificado: Python 3.13, numpy 2.3, pandas 2.2, scikit-learn 1.6.1,
scipy 1.17. A solução K = 3 difere em um participante entre scikit-learn 1.6.1 e
1.8.0 (ARI 0,989); as tabelas do manuscrito usam os rótulos oficiais 91/87/60.


## ⚙️ Como executar

### Pré-requisitos
Instale as dependências executando:
```bash
pip install -r requirements.txt
