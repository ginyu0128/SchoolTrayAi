import streamlit as st

from src.image_io import ImageLoadError, load_uploaded_image
from src.tray_pipeline import estimate_tray_nutrition
from src.cell_splitter import draw_cell_overlay, split_tray_cells
from src.tray_detector import detect_tray, draw_tray_detection
from src.diabetes_advice import build_diabetes_advice
from src.settings import get_setting


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tray-bg: #d8dbe0;
            --tray-surface: #ffffff;
            --tray-ink: #20242d;
            --tray-muted: #6c717a;
            --tray-line: #e4e8e2;
            --tray-teal: #1f6f6a;
            --tray-gold: #c5a25d;
            --tray-marble: #101217;
        }

        .stApp {
            background-color: var(--tray-bg);
            background-image:
                linear-gradient(120deg, rgba(255, 255, 255, 0.72) 0%, rgba(202, 207, 215, 0.48) 44%, rgba(248, 249, 250, 0.64) 100%),
                repeating-linear-gradient(128deg, rgba(18, 21, 27, 0.07) 0 1px, transparent 1px 82px),
                repeating-linear-gradient(36deg, rgba(255, 255, 255, 0.26) 0 1px, transparent 1px 126px),
                linear-gradient(180deg, #eef0f3 0%, #cfd3da 100%);
            color: var(--tray-ink);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }

        .tray-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(197, 162, 93, 0.36);
            background-color: var(--tray-marble);
            background-image:
                linear-gradient(135deg, rgba(255, 255, 255, 0.09) 0%, rgba(255, 255, 255, 0.015) 32%, rgba(197, 162, 93, 0.09) 100%),
                repeating-linear-gradient(112deg, rgba(255, 255, 255, 0.055) 0 1px, transparent 1px 74px),
                repeating-linear-gradient(28deg, rgba(197, 162, 93, 0.06) 0 1px, transparent 1px 128px),
                linear-gradient(180deg, #171a20 0%, #0d0f13 100%);
            box-shadow: 0 24px 70px rgba(11, 12, 15, 0.22);
            border-radius: 8px;
            padding: 2.15rem 2.35rem 2.05rem;
            margin-bottom: 1.35rem;
        }

        .tray-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(8, 9, 12, 0.78) 0%, rgba(8, 9, 12, 0.42) 52%, rgba(8, 9, 12, 0.28) 100%);
            pointer-events: none;
        }

        .tray-hero > * {
            position: relative;
            z-index: 1;
        }

        .tray-kicker {
            color: #d9b869 !important;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.45);
        }

        .tray-title {
            color: #fff8e6 !important;
            font-size: clamp(2.2rem, 4vw, 4.1rem);
            line-height: 1.02;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0;
            text-shadow: 0 3px 18px rgba(0, 0, 0, 0.62);
        }

        .tray-subtitle {
            color: rgba(255, 250, 235, 0.92) !important;
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 760px;
            margin-top: 0.85rem;
            margin-bottom: 0;
            text-shadow: 0 2px 14px rgba(0, 0, 0, 0.5);
        }

        div[data-testid="stFileUploader"] section {
            border: 1px dashed rgba(32, 36, 45, 0.22);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(246, 246, 241, 0.8)),
                repeating-linear-gradient(120deg, rgba(32, 36, 45, 0.025) 0 1px, transparent 1px 74px);
            border-radius: 8px;
        }

        div[data-testid="stFileUploader"] label {
            color: var(--tray-ink);
            font-weight: 750;
        }

        div[data-testid="stImage"] {
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid rgba(32, 36, 45, 0.09);
            box-shadow: 0 14px 34px rgba(32, 36, 45, 0.11);
        }

        h2, h3 {
            color: var(--tray-ink);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(246, 246, 241, 0.82)),
                repeating-linear-gradient(118deg, rgba(32, 36, 45, 0.025) 0 1px, transparent 1px 86px);
            border: 1px solid rgba(32, 36, 45, 0.1);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 28px rgba(32, 36, 45, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--tray-muted);
            font-weight: 750;
        }

        div[data-testid="stMetricValue"] {
            color: var(--tray-ink);
            font-weight: 800;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(32, 36, 45, 0.09);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 12px 34px rgba(32, 36, 45, 0.06);
        }

        div[data-testid="stInfo"] {
            border-radius: 8px;
            border: 1px solid rgba(197, 162, 93, 0.24);
            background:
                linear-gradient(135deg, rgba(246, 248, 245, 0.96), rgba(235, 238, 233, 0.92)),
                repeating-linear-gradient(126deg, rgba(32, 36, 45, 0.025) 0 1px, transparent 1px 82px);
        }

        div[data-testid="stWarning"] {
            border-radius: 8px;
            border: 1px solid rgba(185, 130, 47, 0.2);
        }

        .stCaption {
            color: var(--tray-muted);
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 1rem;
            }

            .tray-hero {
                border-radius: 8px;
                padding: 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="tray-hero">
            <div class="tray-kicker">School Meal Analysis</div>
            <h1 class="tray-title">SchoolTrayAI</h1>
            <p class="tray-subtitle">
                급식판 사진 한 장으로 칸별 음식, 부피, 무게, 칼로리와 영양성분을 추정합니다.
                분석 과정은 이미지 확인부터 영양표와 건강 조언까지 한 화면에서 이어집니다.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="SchoolTrayAI", layout="wide")
inject_page_styles()
render_header()

if get_setting("OPENAI_API_KEY"):
    st.caption("음식 인식: OpenAI Vision 활성화")
else:
    st.warning("OPENAI_API_KEY가 설정되지 않아 AI 음식 인식이 비활성화되어 있습니다.")

if get_setting("FOODSAFETYKOREA_API_KEY") or get_setting("DATA_GO_KR_API_KEY"):
    st.caption("영양정보 API: 공공데이터포털 연동 활성화")
else:
    st.warning("영양정보 API 키가 설정되지 않아 임시 로컬 영양값을 사용합니다.")

uploaded_file = st.file_uploader("급식판 이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image_np = load_uploaded_image(uploaded_file)
    except ImageLoadError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("급식판 칸을 분석하고 음식을 인식하는 중..."):
        results = estimate_tray_nutrition(image_np)
    detection = results.get("tray_detection")
    if detection is None:
        fallback_detection = detect_tray(image_np)
        detection = {
            "bbox": fallback_detection.bbox,
            "found": fallback_detection.found,
            "area_ratio": round(fallback_detection.area_ratio, 4),
        }

    tray_debug_image = results.get("debug_images", {}).get("tray_detection")
    if tray_debug_image is None:
        tray_debug_image = draw_tray_detection(image_np, detect_tray(image_np))

    cell_debug_image = results.get("debug_images", {}).get("cell_split")
    if cell_debug_image is None:
        tray_bbox = tuple(detection["bbox"])
        cell_debug_image = draw_cell_overlay(image_np, split_tray_cells(tray_bbox))

    st.caption(f"Image size: {image_np.shape[1]} x {image_np.shape[0]} px")
    original_col, detected_col, cell_col = st.columns(3)
    with original_col:
        st.image(image_np, caption="업로드 이미지", width="stretch")
    with detected_col:
        st.image(
            tray_debug_image,
            caption="급식판 인식",
            width="stretch",
        )
    with cell_col:
        st.image(
            cell_debug_image,
            caption="칸 분리",
            width="stretch",
        )

    status = "detected" if detection["found"] else "fallback to full image"
    st.caption(
        f"Tray detection: {status}, bbox={detection['bbox']}, "
        f"area_ratio={detection['area_ratio']}"
    )
    flat_piece_cells = results.get("flat_piece_cells", [])
    if flat_piece_cells:
        st.info(f"Flat piece detected: {', '.join(flat_piece_cells)}")
    else:
        st.info("Flat piece detected: none")

    st.subheader("영양 추정 결과")
    nutrition_summary = results["nutrition_summary"]
    total_row = nutrition_summary[nutrition_summary["cell_id"] == "TOTAL"].iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("총 칼로리", f"{float(total_row['calories']):.0f} kcal")
    metric_cols[1].metric("총 무게", f"{float(total_row['weight_g']):.0f} g")
    metric_cols[2].metric("탄수화물", f"{float(total_row['carbs_g']):.1f} g")
    metric_cols[3].metric("단백질 / 지방", f"{float(total_row['protein_g']):.1f} g / {float(total_row['fat_g']):.1f} g")

    nutrition_details = []
    if "nutrition_detail" in nutrition_summary.columns:
        local_rows = nutrition_summary[nutrition_summary["nutrition_source"] == "local_catalog"]
        nutrition_details = [
            str(detail)
            for detail in local_rows["nutrition_detail"].dropna().unique()
            if str(detail).strip() and str(detail).strip() != "-"
        ]

    if nutrition_details:
        st.warning("일부 항목은 영양 API 검색에 실패해 임시값을 사용했습니다.")
        with st.expander("영양 API 진단 메시지", expanded=False):
            st.code("\n\n".join(nutrition_details), language="text")

    display_columns = [
        "cell_id",
        "food_name",
        "nutrition_search_name",
        "volume_profile",
        "weight_g",
        "calories",
        "carbs_g",
        "protein_g",
        "fat_g",
        "nutrition_source",
        "classification_confidence",
        "classification_reason",
        "nutrition_detail",
    ]
    visible_columns = [column for column in display_columns if column in nutrition_summary.columns]
    column_labels = {
        "cell_id": "칸",
        "food_name": "인식 음식명",
        "nutrition_search_name": "영양 검색어",
        "volume_profile": "부피 모델",
        "weight_g": "무게(g)",
        "calories": "칼로리(kcal)",
        "carbs_g": "탄수화물(g)",
        "protein_g": "단백질(g)",
        "fat_g": "지방(g)",
        "nutrition_source": "영양 출처",
        "classification_confidence": "인식 신뢰도",
        "classification_reason": "인식 근거",
        "nutrition_detail": "영양 API 상세",
    }
    st.dataframe(
        nutrition_summary[visible_columns].rename(columns=column_labels),
        width="stretch",
    )

    diabetes_advice = build_diabetes_advice(nutrition_summary)
    st.subheader("당뇨가 걱정된다면?")
    st.info(diabetes_advice["summary"])

    st.subheader("칸별 상세 결과")
    st.json(results["cell_estimates"])
