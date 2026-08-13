import streamlit as st
import requests
import base64
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

st.set_page_config(
    page_title="MNIST CNN Classifier",
    page_icon="🔢",
    layout="centered"
)

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN DASHBOARD CONTAINER
       ======================================================== */

    .block-container {
        max-width: 1000px;

        /* Spacing inside the box */
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;

        /* Outer box */
        border: 1px solid #374151;
        border-radius: 16px;

        /* Make sure border wraps the entire container */
        box-sizing: border-box;

        /* Slight separation from browser edges */
        margin-top: 2rem;
        margin-bottom: 2rem;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }


    /* ========================================================
       SUBTITLE
       ======================================================== */

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 35px;
    }


    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
    }


    /* ========================================================
       PREDICT BUTTON
       ======================================================== */

    div.stButton > button {
        background-color: white;
        color: #111827;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        font-weight: 600;
    }


    div.stButton > button:hover {
        background-color: #e5e7eb;
        color: #111827;
        border-color: #d1d5db;
    }


    /* ========================================================
       RADIO BUTTON
       ======================================================== */

    div[data-testid="stRadio"] label div[role="radio"] {
        border-color: #6b7280 !important;
    }


    /* Selected radio */
    div[data-testid="stRadio"] label div[role="radio"][aria-checked="true"] {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
    }


    /* Inner dot */
    div[data-testid="stRadio"] label div[role="radio"][aria-checked="true"]::after {
        background-color: #111827 !important;
    }


    /* ========================================================
       COLUMN SEPARATOR
       ======================================================== */

    /* Optional: subtle separator between Input and Prediction */
    div[data-testid="column"]:first-child {
        border-right: 1px solid #374151;
        padding-right: 2rem;
    }

    div[data-testid="column"]:last-child {
        padding-left: 1rem;
    }


    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">MNIST CNN Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload a handwritten digit and let the CNN model predict the digit.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

KSERVE_URL = (
    "https://faced-defense-youth-institutions.trycloudflare.com"
    "/v1/models/mnist-cnn:predict"
)
# KSERVE_URL = (
#     "http://127.0.0.1:8080"
#     "/v1/models/mnist-cnn:predict"
# )

left_col, right_col = st.columns(
    2,
    gap="large"
)

with left_col:

    st.markdown(
        '<div class="section-title">Input Digit</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Choose how you want to provide the digit."
    )

    input_method = st.radio(
        "Input method",
        ["Upload Image", "Draw Digit"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if input_method == "Upload Image":

        st.write(
            "Choose a handwritten digit image."
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            help="Supported formats: PNG, JPG, JPEG"
        )


        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            )

            st.image(
                image,
                caption="Uploaded Image",
                width=250
            )


    else:

        st.write(
            "Draw a handwritten digit using your mouse or touchpad."
        )

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_canvas"
        )

with right_col:

    st.markdown(
        '<div class="section-title">Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The prediction result will appear here after you click the button."
    )


    predict_button = st.button(
        "Predict Digit",
        use_container_width=True
    )


    if predict_button:

        try:

            if input_method == "Upload Image":

                if uploaded_file is None:

                    st.warning(
                        "Please upload an image first."
                    )

                    st.stop()


                image_bytes = uploaded_file.getvalue()

            else:

                if canvas_result.image_data is None:

                    st.warning(
                        "Please draw a digit first."
                    )

                    st.stop()


                canvas_image = Image.fromarray(
                    canvas_result.image_data.astype("uint8"),
                    mode="RGBA"
                )


                buffer = io.BytesIO()

                canvas_image.save(
                    buffer,
                    format="PNG"
                )

                image_bytes = buffer.getvalue()


            image_b64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")


            response = requests.post(
                KSERVE_URL,
                json={
                    "instances": [
                        image_b64
                    ]
                },
                timeout=30
            )


            if response.status_code == 200:

                result = response.json()

                prediction = result[
                    "predictions"
                ][0]


                st.write(
                    f"### Prediction : {prediction}"
                )

            else:

                st.error(
                    f"""
                    KServe returned an error.

                    HTTP Status:
                    {response.status_code}

                    Response:
                    {response.text}
                    """
                )

        except requests.exceptions.Timeout:

            st.error(
                "Request to KServe timed out."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                """
                Could not connect to KServe.

                Make sure the KServe port-forward
                is running.
                """
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )