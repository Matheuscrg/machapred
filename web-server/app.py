import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st

#page config

st.set_page_config(page_title="MaChaPred", layout="centered")

# This section talks about the style of the page. This syle of sidebars was inspired by Serotonin-AI
# We are using and external font

def set_ui():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Bitcount:wght@100..900&display=swap" rel="stylesheet">

        <style>
        .stApp {
            background:
                linear-gradient(rgba(255,255,255,0.78), rgba(255,255,255,0.78)),
                url("https://raw.githubusercontent.com/Matheuscrg/machapred/refs/heads/main/images/background.png");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .main-title {
            font-family: 'Bitcount', sans-serif !important;
            font-size: 90px;
            font-weight: 700;
            text-align: center;
            margin-top: 0.2rem;
            margin-bottom: 0.1rem;
            color: #111111;
            line-height: 1;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #333333;
            margin-top: 0;
            margin-bottom: 30px;
        }

        .block-container {
            background: rgba(255,255,255,0.86);
            padding: 2rem;
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_ui()

col_empty, col_lang = st.columns([3, 1])
with col_lang:
    lang = st.radio("", ["EN", "PT"], horizontal=True)

#This section right here is what allow us to change the name of the interface based upon the language

UI = {
    "EN": {
        "subtitle": "A Deep learning app for predicting bioactive compounds against Plasmodium sp. and Trypanossoma cruzi",
        "error_no_models": "Place .pt models in the /models folder",
        "choose_model": "Choose a model",
        "about_models": "📚 About the Models",
        "threshold": "Threshold (minimum 0.70)",
        "paste_smiles": "Paste SMILES (one per line)",
        "run_btn": "Run prediction",
        "clear_btn": "Clear state",
        "state_cleared": "State cleared.",
        "running": "Running Chemprop...",
        "changed_model": "You changed the model. Click 'Run prediction' again.",
        "err_no_col": "Prediction column not found.",
        "prob_col": "Probability column (class 1 = Bioactive)",
        "err_non_numeric": "The column '{}' contains non-numeric values.",
        "bioactive": "Bioactive",
        "not_bioactive": "Not bioactive",
        "completed": "Completed! Model: {} • Bioactive if p ≥ {:.2f}",
        "download": "Download CSV",
        "auto_delete": "The file will be automatically deleted from the server in 5 minutes.",
        "logs": "Chemprop Logs",
        "waiting": "Choose a model, paste the SMILES and click 'Run prediction'.",
        "err_no_smiles": "Please provide at least 1 SMILES.",
        "err_too_many_smiles": "Please provide a maximum of 10 SMILES per run."
    },
    "PT": {
        "subtitle": "Um aplicativo de Deep Learning para predição de compostos bioativos contra Plasmodium sp. e Trypanossoma cruzi",
        "error_no_models": "Coloque modelos .pt na pasta /models",
        "choose_model": "Escolha o modelo",
        "about_models": "📚 Sobre os Modelos",
        "threshold": "Threshold (mínimo 0.70)",
        "paste_smiles": "Cole SMILES (um por linha)",
        "run_btn": "Rodar predição",
        "clear_btn": "Limpar estado",
        "state_cleared": "Estado limpo.",
        "running": "Rodando Chemprop...",
        "changed_model": "Você trocou o modelo. Clique em 'Rodar predição' novamente.",
        "err_no_col": "Não encontrei coluna de predição.",
        "prob_col": "Coluna de probabilidade (classe 1 = Bioativo)",
        "err_non_numeric": "A coluna '{}' tem valores não numéricos.",
        "bioactive": "Bioativo",
        "not_bioactive": "Não bioativo",
        "completed": "Concluído! Modelo: {} • Bioativo se p ≥ {:.2f}",
        "download": "Baixar CSV",
        "auto_delete": "O arquivo será apagado automaticamente do servidor em 5 minutos.",
        "logs": "Logs do Chemprop",
        "waiting": "Escolha um modelo, cole os SMILES e clique em Rodar predição.",
        "err_no_smiles": "Por favor, forneça pelo menos 1 SMILES.",
        "err_too_many_smiles": "Por favor, forneça no máximo 10 SMILES por vez."
    }
}

t = UI[lang] # Active dictionary

#Title

st.markdown(
    f"""
    <h1 class="main-title">MaChaPred</h1>
    <p class="subtitle">
    {t["subtitle"]}
    </p>
    """,
    unsafe_allow_html=True
)

# Here is the place where we defined some backend confs.

MODELS_DIR = Path("models")
RESULTS_DIR = Path("tmp_results")

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DELETE_AFTER_SECONDS = 300 #When the software will delete results
CHEMPROP_TIMEOUT_SECONDS = 300 #When will it stop running if the task is too complicated
STALE_SECONDS = 30 * 60

#UI functions

def cleanup_stale_files(results_dir: Path, stale_seconds: int) -> None:
    now = time.time()
    for p in results_dir.glob("result_*.csv"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > stale_seconds:
                p.unlink()
        except Exception:
            pass

cleanup_stale_files(RESULTS_DIR, STALE_SECONDS)

def schedule_delete(path: Path) -> None:
    def delete():
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    t_thread = threading.Timer(DELETE_AFTER_SECONDS, delete)
    t_thread.daemon = True
    t_thread.start()

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=CHEMPROP_TIMEOUT_SECONDS,
        env=env,
    )

def _extract_smiles(raw_text: str) -> List[str]:
    smiles = [s.strip() for s in raw_text.splitlines() if s.strip()]
    if len(smiles) == 0:
        raise ValueError(t["err_no_smiles"])
    if len(smiles) > 10: #Here you can impose limitations on the server running, if you do not need this limitation just remove this if
        raise ValueError(t["err_too_many_smiles"])
    return smiles

def list_models() -> List[Path]:
    return sorted(
        [p for p in MODELS_DIR.glob("*.pt") if p.is_file()],
        key=lambda p: p.name.lower()
    )

def chemprop_predict(model: Path, smiles: List[str]) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame({"smiles": smiles})

    with tempfile.TemporaryDirectory() as td:
        in_path = f"{td}/input.csv"
        out_path = f"{td}/preds.csv"

        df.to_csv(in_path, index=False)

        cmd = [
            "chemprop", "predict",
            "--test-path", in_path,
            "--model-path", str(model),
            "--preds-path", out_path
        ]

        result = _run(cmd)
        logs = f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

        if result.returncode != 0:
            raise RuntimeError(logs)

        return pd.read_csv(out_path), logs


# Here is where the input is mananged 

models = list_models()

if not models:
    st.error(t["error_no_models"])
    st.stop()

model_names = [m.stem for m in models]
selected_name = st.selectbox(t["choose_model"], model_names)
selected_model = MODELS_DIR / f"{selected_name}.pt"

# Descriptions of the models used 

with st.expander(t["about_models"], expanded=False):
    if lang == "EN":
        st.markdown("""
        **T_cruzi_org**
        This generalist model uses all available IC₅₀ data for Chagas in the CheMBL database. The objective is to generate generalist predictions of anti-chagasic activity using a myriad of available experiments.

        **T_cruzi_Amastigote**
        The T_cruzi_Amastigote is a model derived from the T_cruzi_model, with a filter applied to retain experiments performed in the amastigote stage. The amastigotes are generally recognized as an intracellular, replicating form of the parasite, and sometimes these amastigotes may reach a dormant stage that no longer replicates, has even absent metabolic activity, and is especially resistant to drug treatments. Furthermore, drug development efforts should primarily focus on this specific stage of the parasite, as it would be a better use of time and resources.

        **T_cruzi_Trypomastigote**
        The t_cruzi_trypomastigote was developed specifically to address inhibitors targeting the parasite's Trypomastigote stage. This model is particularly relevant because trypomastigotes invade host cells and are found in the host's bloodstream. Furthermore, many research groups find it difficult to cultivate parasites at this stage as trypomastigotes neither multiply nor survive long enough in vitro and can, in some cases, differentiate back into the amastigote form. Therefore, it would be beneficial to researchers to have this initial screening using AI. Additionally, it is worth noting that metacyclic trypomastigotes are the infective form that enters the human body during the initial stages of infection.

        **T_cruzi_epimastigote**
        This model was made using keywords related to epimastigote assays in the original T. cruzi dataset downloaded from CHEMBL 36, as detailed on Table S2. The biological rationale for this model is that this specific life form is found in the gut of insect vectors. Because they are considered easy to cultivate, they are widely used by researchers today; however, there are criticisms of the translational value of these results, as this form is not present in mammalian hosts.

        **T_cruzi_Sterol_Demthylase**
        The model T_cruzi_Sterol_Demethylase is based on a target enzyme from the ChemBL database. This enzyme is a potential drug target and has shown good activity in preclinical models. The biological basis of this enzyme inhibition is that it can be lethal in organisms that require sterol biosynthesis, such as T. cruzi. Also, studies have demonstrated that inhibitors of this enzyme damage the parasites' membranes, with an even stronger effect in amastigote forms.

        **T_cruzi_Cruzipain**
        Cruzipain is a highly glycosylated cystein protease that is considered to be one of the main molecular targets for drug development against Chagas disease. It is also the most abundant protein of the class protease in T. cruzi, and it is responsible for diverse biological activities such as evasion of the host immune system and degradation of important proteins that trigger relevant cellular processes; thus, the inhibition of this protein may pose intense difficulties for the parasite’s survival. Even though this model was built on a small dataset, it remains biologically relevant.

        **Plasmodium_falciparum_ORG**
        This model is built using data from P. falciparum, one of the main etiological agents of malaria, which undergoes a complex life cycle that includes an intracellular phase and an erythrocyte phase. This is a model built on a large dataset that includes a myriad of malaria-related experiments (resistant and non-resistant), so it is not specific.

        **Blood_stage_Malaria**
        This model was made from a subset of the Plasmodium_falciparum_ORG dataset, applying keywords related to blood/erythrocytes, as available on S2. The development of treatments for the blood stage of malaria, also known as the erythrocytic stage, is one of the main goals of malaria drug development, as this stage is responsible for the pathogenesis and clinical manifestations of malaria. However, it is important to note that ChEMBL MAIP is also a valuable tool for predicting blood-stage malaria drugs.

        **P_berghei_ORG**
        Unlike P. falciparum and P. vivax, P. berghei does not naturally cause malaria in humans; however, it has been widely used as a model organism to study pathophysiology, the host’s immunological response, and treatment targets in murine models. This model was built to be a generalist model, like P_faciparum_ORG and T_cruzi_ORG, as an aid in the discovery of new bioactive drugs related to murine models.

        **CQ_resistant_P_falciparum**
        This model was built using a subset of the P. falciparum dataset, with keywords related to chloroquine resistance Table S2. Antimalarial drug resistance and declining efficacy remain major problems for science, so this model targets strains that carry resistance to be a more precise predictor of the emerging problems.
        """)
    else:
        st.markdown("""
        **T_cruzi_org**
        Este modelo generalista usa todos os dados de IC₅₀ disponíveis para Chagas no banco de dados CheMBL. O objetivo é gerar predições generalistas de atividade antichagásica usando uma miríade de experimentos disponíveis.

        **T_cruzi_Amastigote**
        O T_cruzi_Amastigote é um modelo derivado do T_cruzi_model, com um filtro aplicado para reter experimentos realizados no estágio amastigota. As amastigotas são geralmente reconhecidas como uma forma intracelular de replicação do parasita, e às vezes podem atingir um estágio dormente que não se replica mais, apresentando até ausência de atividade metabólica, sendo especialmente resistentes a tratamentos. O desenvolvimento de medicamentos deve focar principalmente neste estágio, pois seria um melhor uso de tempo e recursos.

        **T_cruzi_Trypomastigote**
        O t_cruzi_trypomastigote foi desenvolvido especificamente para focar em inibidores do estágio Tripomastigota do parasita. Este modelo é relevante porque os tripomastigotas invadem as células do hospedeiro e são encontrados na corrente sanguínea. Além disso, muitos grupos de pesquisa têm dificuldade em cultivar parasitas nesta fase, pois eles não se multiplicam nem sobrevivem por muito tempo in vitro, podendo, em alguns casos, se diferenciar de volta na forma amastigota. Ter uma triagem inicial usando IA é altamente benéfico.

        **T_cruzi_epimastigote**
        Este modelo foi construído usando palavras-chave relacionadas a ensaios de epimastigotas no conjunto original do T. cruzi. A justificativa biológica é que esta forma de vida específica é encontrada no intestino de insetos vetores. Por serem considerados fáceis de cultivar, são amplamente utilizados; no entanto, existem críticas quanto ao valor translacional destes resultados, uma vez que esta forma não está presente em hospedeiros mamíferos.

        **T_cruzi_Sterol_Demthylase**
        O modelo baseia-se em uma enzima alvo do banco de dados ChemBL. Esta enzima é um potencial alvo farmacológico com boa atividade em modelos pré-clínicos. A base biológica da sua inibição é que ela pode ser letal em organismos que requerem a biossíntese de esteróis. Estudos demonstraram que inibidores desta enzima danificam as membranas dos parasitas, com um efeito ainda mais forte em formas amastigotas.

        **T_cruzi_Cruzipain**
        A cruzipaína é uma protease de cisteína altamente glicosilada, considerada um dos principais alvos moleculares para o desenvolvimento de medicamentos contra a doença de Chagas. É a proteína mais abundante da sua classe no T. cruzi e é responsável por diversas atividades biológicas (como evasão do sistema imune e degradação de proteínas). A inibição desta proteína pode criar intensas dificuldades para a sobrevivência do parasita.

        **Plasmodium_falciparum_ORG**
        Este modelo é construído a partir de dados do P. falciparum, um dos principais agentes etiológicos da malária, que possui um ciclo de vida complexo (fases intracelular e eritrocítica). É um modelo construído sobre um grande conjunto de dados que inclui uma miríade de experimentos relacionados à malária (resistentes e não resistentes), portanto não é específico.

        **Blood_stage_Malaria**
        Este modelo foi feito a partir de um subconjunto de dados do P. falciparum, aplicando palavras-chave relacionadas a sangue/eritrócitos. O desenvolvimento de tratamentos para o estágio sanguíneo da malária (estágio eritrocítico) é um dos principais objetivos farmacológicos, pois é o responsável pela patogênese e manifestações clínicas.

        **P_berghei_ORG**
        Ao contrário do P. falciparum e P. vivax, o P. berghei não causa malária naturalmente em humanos; entretanto, tem sido amplamente utilizado como organismo modelo para estudar fisiopatologia, resposta imunológica e alvos de tratamento em modelos murinos. Foi construído para ser um modelo generalista para auxiliar na descoberta de novos medicamentos bioativos relacionados a estes modelos murinos.

        **CQ_resistant_P_falciparum**
        Construído usando um subconjunto de dados do P. falciparum com foco em resistência à cloroquina. A resistência aos medicamentos antimaláricos e o declínio de sua eficácia continuam sendo grandes problemas para a ciência, de modo que este modelo foca em cepas resistentes para ser um preditor mais preciso destes problemas emergentes.
        """)

# Here is the rest of the user interface

threshold = st.slider(t["threshold"], 0.70, 1.0, 0.70, 0.01)

raw = st.text_area(
    t["paste_smiles"],
    height=180,
    placeholder="CCO\nc1ccccc1\nCC(=O)O"
)

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button(t["run_btn"], type="primary")
with col2:
    clear_btn = st.button(t["clear_btn"])

#Here what happens when the clear button is pressed

if clear_btn:
    for k in ["preds", "logs", "model_name", "smiles_used"]:
        st.session_state.pop(k, None)
    st.success(t["state_cleared"])
    
#Here what happens when the run button is pressed

if run_btn:
    try:
        smiles = _extract_smiles(raw)

        with st.spinner(t["running"]):
            preds, logs = chemprop_predict(selected_model, smiles)

        st.session_state["preds"] = preds
        st.session_state["logs"] = logs
        st.session_state["model_name"] = selected_name
        st.session_state["smiles_used"] = smiles

    except Exception as e:
        st.error(str(e))
        st.stop()

if "preds" in st.session_state:
    preds = st.session_state["preds"]

    if st.session_state.get("model_name") != selected_name:
        st.warning(t["changed_model"])

    prob_cols = [c for c in preds.columns if c.lower() != "smiles"]
    if not prob_cols:
        st.error(t["err_no_col"])
        st.stop()

    prob_col = prob_cols[0] if len(prob_cols) == 1 else st.selectbox(
        t["prob_col"],
        prob_cols,
        index=0
    )

    preds[prob_col] = pd.to_numeric(preds[prob_col], errors="coerce")
    if preds[prob_col].isna().any():
        st.error(t["err_non_numeric"].format(prob_col))
        st.stop()

    preds_out = pd.DataFrame()
    preds_out["smiles"] = preds["smiles"] if "smiles" in preds.columns else st.session_state.get("smiles_used", [])
    preds_out["p_bioactive"] = preds[prob_col].astype(float)
    preds_out["class"] = (preds_out["p_bioactive"] >= threshold).astype(int)
    preds_out["prediction"] = preds_out["class"].map({1: t["bioactive"], 0: t["not_bioactive"]})

    st.success(t["completed"].format(selected_name, threshold))
    st.dataframe(preds_out, use_container_width=True)

    path = RESULTS_DIR / f"result_{int(time.time())}.csv"
    preds_out.to_csv(path, index=False)
    schedule_delete(path)

    st.download_button(
        t["download"],
        data=path.read_bytes(),
        file_name="result.csv",
        mime="text/csv"
    )

    st.info(t["auto_delete"])

    with st.expander(t["logs"]):
        st.code(st.session_state.get("logs", ""), language="text")

else:
    st.info(t["waiting"])
