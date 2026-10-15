# oss-matchmaker-ml

Recomenda repositórios e issues de um catálogo sintético por similaridade textual; aceita histórico de interações para misturar sinais colaborativos.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: avaliação de recuperação por oito perfis sintéticos com temas conhecidos sobre catálogo de 24 itens — precision@5 **0,6000**, recall@5 **1,0000**. O recomendador usa TF-IDF por conteúdo; quando o input contém histórico suficiente para comparar usuários, mistura um sinal colaborativo simples. Os scores são medidos em um catálogo artificial e não predizem relevância real no GitHub.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
