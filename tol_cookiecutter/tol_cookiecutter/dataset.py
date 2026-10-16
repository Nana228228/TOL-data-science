"""Pipeline de preparação de dados do TOL19.

Estágios (executar nesta ordem):

    raw (.xlsx/.csv wide)  --restructure-->  interim (long)  --process-->  processed (long + duração)

Cada estágio é um comando Typer independente, mas o fluxo canônico é
`restructure` seguido de `process`. O comando `main` executa os dois em sequência.

Uso:
    python -m tol_cookiecutter.dataset restructure
    python -m tol_cookiecutter.dataset process
    python -m tol_cookiecutter.dataset main        # roda os dois na ordem
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from tol_cookiecutter.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Colunas demográficas fixas por participante (primeiras 7 colunas do raw).
DEMO_COLS = [
    "ID",
    "Sexo",
    "Data de nascimento",
    "Idade",
    "Escolaridade",
    "Estado onde está ocorrendo a aplicação",
    "Data de aplicação",
]

# Cada bloco de tentativa tem 9 colunas, repetidas no formato wide.
TENTATIVE_COLS = [
    "Rodada",
    "Tentativa",
    "Pontuação",
    "Tempo de início",
    "Tempo de fim",
    "Número de movimentos",
    "Descrição dos movimentos",
    "Tempo do primeiro clique correto",
    "Tempo do primeiro clique errado",
]
BLOCK_SIZE = len(TENTATIVE_COLS)

# Colunas inteiras após a reestruturação (nomes ainda em PT, antes da tradução).
INT_COLS = ["Rodada", "Tentativa", "Pontuação", "Número de movimentos"]

# Tradução dos nomes das colunas (PT -> EN), aplicada ao final da reestruturação.
COLUMN_TRANSLATION = {
    "ID": "id",
    "Sexo": "sex",
    "Data de nascimento": "birth_date",
    "Idade": "age",
    "Escolaridade": "education",
    "Estado onde está ocorrendo a aplicação": "state",
    "Data de aplicação": "application_date",
    "Rodada": "round",
    "Tentativa": "attempt",
    "Pontuação": "score",
    "Tempo de início": "start_time",
    "Tempo de fim": "end_time",
    "Número de movimentos": "n_moves",
    "Descrição dos movimentos": "moves_description",
    "Tempo do primeiro clique correto": "first_correct_click_time",
    "Tempo do primeiro clique errado": "first_wrong_click_time",
}

# Tradução dos valores categóricos (PT -> EN).
SEX_TRANSLATION = {
    "Feminino": "Female",
    "Masculino": "Male",
}

EDUCATION_TRANSLATION = {
    "Não Alfabetizado": "Illiterate",
    "Ensino Fundamental Incompleto": "Incomplete Elementary School",
    "Ensino Fundamental Completo": "Complete Elementary School",
    "Ensino Médio Incompleto": "Incomplete High School",
    "Ensino Médio Completo": "Complete High School",
    "Técnico/Profissionalizante Incompleto": "Incomplete Technical/Vocational",
    "Técnico/Profissionalizante Completo": "Complete Technical/Vocational",
    "Ensino Superior Incompleto": "Incomplete Higher Education",
    "Ensino Superior Completo": "Complete Higher Education",
    "Pós-Graduação: Especialização": "Graduate: Specialization",
    "Pós-Graduação: Mestrado": "Graduate: Master",
    "Pós-Graduação: Doutorado": "Graduate: Doctorate",
}

# Colunas de tempo após a tradução (nomes em EN), armazenadas como texto com vírgula decimal.
TIME_COLS = [
    "start_time",
    "end_time",
    "first_correct_click_time",
    "first_wrong_click_time",
]


def _read_any(path: Path) -> pd.DataFrame:
    """Lê .xlsx ou .csv. Para CSV usa separador ',' e vírgula decimal nos tempos."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=",", encoding="utf-8-sig", low_memory=False)


def restructure_wide_to_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transforma o formato wide (blocos de tentativa em colunas) em long (uma linha por tentativa).

    Se a base já estiver no formato long (7 fixas + 9 por tentativa = 16 colunas),
    apenas renomeia e segue sem reestruturar.
    """
    if df_raw.shape[1] == len(DEMO_COLS) + BLOCK_SIZE:
        logger.info("Base já está no formato long; reestruturação não é necessária.")
        df = df_raw.copy()
        df.columns = DEMO_COLS + TENTATIVE_COLS
    else:
        n_blocks = (df_raw.shape[1] - len(DEMO_COLS)) // BLOCK_SIZE
        logger.info(f"Formato wide detectado: {n_blocks} blocos de tentativa por participante.")

        demo_values = df_raw.iloc[:, : len(DEMO_COLS)].values
        trial_values = df_raw.iloc[
            :, len(DEMO_COLS) : len(DEMO_COLS) + n_blocks * BLOCK_SIZE
        ].values

        trial_reshaped = trial_values.reshape(len(df_raw), n_blocks, BLOCK_SIZE)
        demo_repeated = np.repeat(demo_values, n_blocks, axis=0)
        trials_flat = trial_reshaped.reshape(-1, BLOCK_SIZE)

        df = pd.DataFrame(
            np.column_stack([demo_repeated, trials_flat]),
            columns=DEMO_COLS + TENTATIVE_COLS,
        )

    # Remove linhas sem Rodada (participantes com menos tentativas).
    df = df.dropna(subset=["Rodada"]).reset_index(drop=True)

    # Converte colunas discretas para inteiro (ainda com nomes em PT).
    for col in INT_COLS:
        df[col] = df[col].astype(int)

    # Traduz nomes de colunas e valores categóricos para o inglês.
    df = df.rename(columns=COLUMN_TRANSLATION)
    df["sex"] = df["sex"].map(SEX_TRANSLATION).fillna(df["sex"])
    df["education"] = df["education"].map(EDUCATION_TRANSLATION).fillna(df["education"])

    return df


def convert_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte os tempos de texto (vírgula decimal) para float e calcula a duração em segundos."""
    df = df.copy()
    for col in TIME_COLS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # start_time/end_time são timestamps em milissegundos; a duração vai para segundos.
    df["duration_seconds"] = (df["end_time"] - df["start_time"]) / 1000

    # Durações impossíveis (negativas ou acima de 24h) viram NaN, sem descartar a linha.
    mascara_invalida = (df["duration_seconds"] < 0) | (df["duration_seconds"] > 24 * 60 * 60)
    n_invalidas = int(mascara_invalida.sum())
    if n_invalidas:
        logger.warning(f"{n_invalidas} durações inválidas convertidas para NaN.")
    df.loc[mascara_invalida, "duration_seconds"] = np.nan

    return df


@app.command()
def restructure(
    input_path: Path = RAW_DATA_DIR / "1752256927-TOL19-tests.xlsx",
    output_path: Path = INTERIM_DATA_DIR / "tol_long.csv",
):
    """Estágio 1: raw (wide) -> interim (long)."""
    logger.info(f"Lendo base bruta: {input_path}")
    df_raw = _read_any(input_path)
    logger.info(f"Shape bruto: {df_raw.shape}")

    df = restructure_wide_to_long(df_raw)
    logger.info(f"Shape long: {df.shape} | participantes: {df['id'].nunique():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.success(f"Base long salva em: {output_path}")


@app.command()
def process(
    input_path: Path = INTERIM_DATA_DIR / "tol_long.csv",
    output_path: Path = PROCESSED_DATA_DIR / "tol_long_diagnosticado.csv",
):
    """Estágio 2: interim (long) -> processed (tempos convertidos + duração)."""
    logger.info(f"Lendo base intermediária: {input_path}")
    df = _read_any(input_path)
    logger.info(f"Shape de entrada: {df.shape}")

    df = convert_time_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.success(f"Base processada salva em: {output_path} | shape: {df.shape}")


@app.command()
def main(
    raw_path: Path = RAW_DATA_DIR / "1752256927-TOL19-tests.xlsx",
    interim_path: Path = INTERIM_DATA_DIR / "tol_long.csv",
    processed_path: Path = PROCESSED_DATA_DIR / "tol_long_diagnosticado.csv",
):
    """Pipeline completa: executa restructure e process na ordem."""
    restructure(input_path=raw_path, output_path=interim_path)
    process(input_path=interim_path, output_path=processed_path)
    logger.success("Pipeline de dados concluída.")


if __name__ == "__main__":
    app()
