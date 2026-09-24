import re

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import errors, types


# ============================================================
# GEMINI SETTINGS
# ============================================================

PRIMARY_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

MAX_RESUME_CHARS = 12000
MAX_JOB_CHARS = 12000


# ============================================================
# GEMINI HELPER
# ============================================================

def generate_gemini_response(prompt, max_output_tokens=800):
    """
    Send a prompt to Gemini.

    Primary model is tried first.
    Fallback model is used for temporary API errors.
    """

    client = genai.Client()

    try:

        response = client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens
            )
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text

    except errors.APIError as first_error:

        # Retry only for temporary errors
        if first_error.code not in (429, 500, 503, 504):
            raise first_error

        try:

            response = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens
                )
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text

        except errors.APIError as second_error:

            raise RuntimeError(
                f"Gemini request failed.\n"
                f"Primary model error: {first_error.code}\n"
                f"Fallback model error: {second_error.code}\n"
                f"Please try again later."
            ) from second_error


# ============================================================
# GEMINI FUNCTIONS
# ============================================================

def generate_ai_questions(resume_text, job_description):

    resume_text = resume_text[:MAX_RESUME_CHARS]
    job_description = job_description[:MAX_JOB_CHARS]

    prompt = f"""
You are an interview preparation assistant.

Create exactly 5 interview questions for a student applying
for the job below.

Use BOTH:
1. The student's resume
2. The job description

The questions should test:

- Technical skills required by the job
- Skills mentioned in the resume
- Projects or experience mentioned in the resume
- One question about a skill the student is missing

Job Description:
{job_description}

Resume:
{resume_text}

Return ONLY the 5 questions as a numbered list.
Do not include explanations or answers.
"""

    return generate_gemini_response(
        prompt,
        max_output_tokens=600
    )


def evaluate_answer(
    question,
    answer,
    resume_text,
    job_description
):

    resume_text = resume_text[:MAX_RESUME_CHARS]
    job_description = job_description[:MAX_JOB_CHARS]

    prompt = f"""
You are an interview coach helping a student prepare
for a job interview.

Evaluate the student's answer to the interview question.

Interview Question:
{question}

Student's Answer:
{answer}

Job Description:
{job_description}

Resume:
{resume_text}

Give feedback using exactly these sections:

1. What you did well
2. What could be improved
3. Important points you missed
4. A stronger example answer

Be constructive and specific.

Do not claim that this is an actual hiring decision.
"""

    return generate_gemini_response(
        prompt,
        max_output_tokens=900
    )


def generate_skill_recommendations(
    missing_skills,
    job_description
):

    skills_text = ", ".join(missing_skills)

    job_description = job_description[:MAX_JOB_CHARS]

    prompt = f"""
You are a career guidance assistant.

The student is applying for this job:

Job Description:
{job_description}

The following skills were detected as missing
from the student's resume:

{skills_text}

For EACH missing skill:

1. Explain briefly why it matters for this job.
2. Give 2-3 specific things the student should learn or practice.
3. Keep the advice practical for a college student.

Return the result as a clear numbered list.
"""

    return generate_gemini_response(
        prompt,
        max_output_tokens=900
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def contains_skill(text, skill):
    """
    Check whether a skill appears as a real word/phrase.
    """

    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

    return re.search(
        pattern,
        text,
        re.IGNORECASE
    ) is not None


def contains_c_skill(text):
    """
    Special handling for C because searching for the
    standalone letter 'c' would create loads of false matches.
    """

    c_patterns = [
        r"\bc programming\b",
        r"\bc language\b",
        r"\bprogramming in c\b",
        r"\bc developer\b",
        r"\bc development\b",
        r"\busing c\b",
        r"\bc\s+programming\b",
        r"\bansi c\b"
    ]

    for pattern in c_patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            return True

    return False


def clean_resume_for_ai(resume_text):
    """
    Remove unnecessary contact information before sending
    resume text to Gemini.
    """

    # Remove email addresses
    resume_text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REMOVED]",
        resume_text
    )

    # Remove phone numbers
    resume_text = re.sub(
        r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)",
        "[PHONE REMOVED]",
        resume_text
    )

    # Remove URLs
    resume_text = re.sub(
        r"https?://\S+",
        "[LINK REMOVED]",
        resume_text
    )

    return resume_text


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="HireLens",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔎 HireLens")

st.subheader(
    "See the gap. Sharpen the skill. Nail the interview."
)

st.write(
    "Upload your resume and paste a job description "
    "to discover your skill match and prepare for the interview."
)

st.divider()


# ============================================================
# SKILLS
# ============================================================

skill_patterns = {

    "python": [
        "python"
    ],

    "java": [
        "java"
    ],

    "c++": [
        "c++",
        "cpp"
    ],

    "c#": [
        "c#",
        "c sharp"
    ],

    "javascript": [
        "javascript",
        "java script"
    ],

    "typescript": [
        "typescript"
    ],

    "html": [
        "html",
        "html5"
    ],

    "css": [
        "css",
        "css3"
    ],

    "react": [
        "react",
        "react.js",
        "reactjs"
    ],

    "angular": [
        "angular"
    ],

    "vue": [
        "vue",
        "vue.js",
        "vuejs"
    ],

    "node.js": [
        "node.js",
        "nodejs",
        "node js"
    ],

    "express": [
        "express",
        "express.js",
        "expressjs"
    ],

    "django": [
        "django"
    ],

    "flask": [
        "flask"
    ],

    "sql": [
        "sql"
    ],

    "mysql": [
        "mysql"
    ],

    "postgresql": [
        "postgresql",
        "postgres"
    ],

    "mongodb": [
        "mongodb",
        "mongo db"
    ],

    "sqlite": [
        "sqlite"
    ],

    "pandas": [
        "pandas"
    ],

    "numpy": [
        "numpy"
    ],

    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "machine learning": [
        "machine learning",
        "machine-learning"
    ],

    "deep learning": [
        "deep learning",
        "deep-learning"
    ],

    "data science": [
        "data science",
        "data-science"
    ],

    "artificial intelligence": [
        "artificial intelligence",
        "artificial intelligence (ai)",
        "ai"
    ],

    "git": [
        "git"
    ],

    "github": [
        "github",
        "github.com"
    ],

    "docker": [
        "docker"
    ],

    "kubernetes": [
        "kubernetes",
        "k8s"
    ],

    "aws": [
        "aws",
        "amazon web services"
    ],

    "azure": [
        "azure",
        "microsoft azure"
    ],

    "google cloud": [
        "google cloud",
        "gcp",
        "google cloud platform"
    ],

    "rest api": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
        "restful services"
    ],

    "streamlit": [
        "streamlit"
    ],

    "excel": [
        "excel",
        "microsoft excel"
    ],

    "power bi": [
        "power bi",
        "powerbi"
    ],

    "tableau": [
        "tableau"
    ]
}


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "analysis_done": False,

    "required_skills": [],

    "matched_skills": [],

    "missing_skills": [],

    "resume_text": "",

    "match_score": None,

    "current_question": 0,

    "answer_feedback": "",

    "ai_questions": "",

    "skill_recommendations": ""

}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# RESUME
# ============================================================

st.header("📄 Your Resume")

resume = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("💼 Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=250,
    placeholder=(
        "Paste the full job description here, "
        "not just the job title..."
    )
)


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🚀 Analyze",
    use_container_width=True
):

    if resume is None:

        st.warning(
            "Please upload your resume."
        )

    elif not job_description.strip():

        st.warning(
            "Please paste a job description."
        )

    else:

        # ----------------------------------------------------
        # Read PDF
        # ----------------------------------------------------

        try:

            reader = PdfReader(resume)

            resume_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    resume_text += text + "\n"

        except Exception as e:

            st.error(
                "HireLens could not read this PDF."
            )

            st.code(str(e))

            st.stop()

        # ----------------------------------------------------
        # Check extracted text
        # ----------------------------------------------------

        if not resume_text.strip():

            st.error(
                "No readable text was found in this PDF. "
                "Please upload a text-based PDF resume."
            )

            st.stop()

        # ----------------------------------------------------
        # Convert to lowercase
        # ----------------------------------------------------

        resume_text_lower = resume_text.lower()

        job_text = job_description.lower()

        # ----------------------------------------------------
        # Detect resume skills
        # ----------------------------------------------------

        resume_skills = []

        # Special C detection
        if contains_c_skill(resume_text_lower):

            resume_skills.append("c")

        for skill, patterns in skill_patterns.items():

            for pattern in patterns:

                if contains_skill(
                    resume_text_lower,
                    pattern
                ):

                    resume_skills.append(skill)

                    break

        # ----------------------------------------------------
        # Detect job skills
        # ----------------------------------------------------

        required_skills = []

        # Special C detection
        if contains_c_skill(job_text):

            required_skills.append("c")

        for skill, patterns in skill_patterns.items():

            for pattern in patterns:

                if contains_skill(
                    job_text,
                    pattern
                ):

                    required_skills.append(skill)

                    break

        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

        resume_skills = list(
            dict.fromkeys(resume_skills)
        )

        required_skills = list(
            dict.fromkeys(required_skills)
        )

        # ----------------------------------------------------
        # Matched skills
        # ----------------------------------------------------

        matched_skills = []

        for skill in required_skills:

            if skill in resume_skills:

                matched_skills.append(skill)

        # ----------------------------------------------------
        # Missing skills
        # ----------------------------------------------------

        missing_skills = []

        for skill in required_skills:

            if skill not in resume_skills:

                missing_skills.append(skill)

        # ----------------------------------------------------
        # Match score
        # ----------------------------------------------------

        if required_skills:

            match_score = (
                len(matched_skills)
                / len(required_skills)
            ) * 100

        else:

            match_score = None

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        st.session_state.analysis_done = True

        st.session_state.required_skills = (
            required_skills
        )

        st.session_state.matched_skills = (
            matched_skills
        )

        st.session_state.missing_skills = (
            missing_skills
        )

        st.session_state.match_score = (
            match_score
        )

        st.session_state.resume_text = (
            resume_text
        )

        # Reset AI results

        st.session_state.ai_questions = ""

        st.session_state.skill_recommendations = ""

        st.session_state.current_question = 0

        st.session_state.answer_feedback = ""


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done:

    st.divider()

    st.header("🎯 Your Match Score")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Match score
    # --------------------------------------------------------

    with col1:

        if st.session_state.match_score is not None:

            st.metric(
                "Job Match",
                f"{st.session_state.match_score:.0f}%"
            )

        else:

            st.metric(
                "Job Match",
                "N/A"
            )

    # --------------------------------------------------------
    # Matched skills
    # --------------------------------------------------------

    with col2:

        st.metric(
            "Matched Skills",
            len(
                st.session_state.matched_skills
            )
        )

    # --------------------------------------------------------
    # Missing skills
    # --------------------------------------------------------

    with col3:

        st.metric(
            "Skills to Improve",
            len(
                st.session_state.missing_skills
            )
        )

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    if st.session_state.match_score is not None:

        st.progress(
            int(
                st.session_state.match_score
            ),
            text=(
                f"Overall Match: "
                f"{st.session_state.match_score:.0f}%"
            )
        )

    else:

        st.info(
            "No recognizable skills were detected "
            "in the job description. Please paste "
            "the full job description instead of "
            "only the job title."
        )

    # ========================================================
    # MATCHED / MISSING SKILLS
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Matched
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "✅ Matched Skills"
        )

        if st.session_state.matched_skills:

            for skill in (
                st.session_state.matched_skills
            ):

                st.success(
                    skill.title()
                )

        else:

            st.info(
                "No matching skills found."
            )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "❌ Missing Skills"
        )

        if st.session_state.missing_skills:

            for skill in (
                st.session_state.missing_skills
            ):

                st.error(
                    skill.title()
                )

        elif st.session_state.required_skills:

            st.success(
                "No missing skills! 🎉"
            )

        else:

            st.info(
                "No job skills were detected."
            )

    # ========================================================
    # RECOMMENDED SKILLS
    # ========================================================

    st.divider()

    st.header(
        "📚 Recommended Skills"
    )

    if st.session_state.missing_skills:

        st.write(
            "Skills you may want to improve:"
        )

        for skill in (
            st.session_state.missing_skills
        ):

            st.write(
                f"👉 **{skill.title()}**"
            )

    elif st.session_state.required_skills:

        st.success(
            "Your resume covers all detected skills!"
        )

    else:

        st.info(
            "No skills were detected from the "
            "job description."
        )

    # ========================================================
    # AI SKILL GAP GUIDANCE
    # ========================================================

    if st.session_state.missing_skills:

        if st.button(
            "🤖 Explain My Skill Gaps",
            use_container_width=True
        ):

            with st.spinner(
                "Gemini is analyzing your skill gaps..."
            ):

                try:

                    recommendations = (
                        generate_skill_recommendations(
                            st.session_state.missing_skills,
                            job_description
                        )
                    )

                    st.session_state.skill_recommendations = (
                        recommendations
                    )

                except Exception as e:

                    st.error(
                        "Gemini could not generate "
                        "skill-gap guidance right now."
                    )

                    st.info(
                        "Please try again in a moment."
                    )

                    st.code(str(e))

        # ----------------------------------------------------
        # Display recommendations
        # ----------------------------------------------------

        if st.session_state.skill_recommendations:

            st.subheader(
                "💡 Gemini's Skill Gap Guidance"
            )

            st.markdown(
                st.session_state.skill_recommendations
            )

    # ========================================================
    # INTERVIEW PREPARATION
    # ========================================================

    st.divider()

    st.header(
        "🎤 AI Interview Preparation"
    )

    st.write(
        "Generate interview questions based on "
        "your resume and this job description."
    )

    if st.button(
        "🎯 Generate Interview Questions",
        use_container_width=True
    ):

        with st.spinner(
            "Gemini is creating your interview questions..."
        ):

            try:

                clean_resume = clean_resume_for_ai(
                    st.session_state.resume_text
                )

                questions = generate_ai_questions(
                    clean_resume,
                    job_description
                )

                st.session_state.ai_questions = (
                    questions
                )

                st.session_state.current_question = 0

                st.session_state.answer_feedback = ""

            except Exception as e:

                st.error(
                    "Gemini could not generate "
                    "interview questions."
                )

                st.info(
                    "Please check your Gemini API key "
                    "and try again."
                )

                st.code(str(e))

    # ========================================================
    # DISPLAY INTERVIEW QUESTIONS
    # ========================================================

    if st.session_state.ai_questions:

        st.subheader(
            "📝 Interview Questions"
        )

        st.markdown(
            st.session_state.ai_questions
        )

        st.divider()

        st.subheader(
            "🧑‍💻 Practice Your Answer"
        )

        answer = st.text_area(
            "Write your answer here:",
            height=180,
            placeholder=(
                "Type how you would answer "
                "the interview question..."
            )
        )

        question_number = st.number_input(
            "Question number",
            min_value=1,
            max_value=5,
            value=1,
            step=1
        )

        if st.button(
            "🤖 Evaluate My Answer",
            use_container_width=True
        ):

            if not answer.strip():

                st.warning(
                    "Please write an answer first."
                )

            else:

                # Try to extract numbered questions
                question_matches = re.findall(
                    r"(?:^|\n)\s*\d+[.)]\s*(.+)",
                    st.session_state.ai_questions
                )

                if question_matches and len(
                    question_matches
                ) >= question_number:

                    selected_question = (
                        question_matches[
                            question_number - 1
                        ]
                    )

                else:

                    selected_question = (
                        "Interview question "
                        f"{question_number}"
                    )

                with st.spinner(
                    "Gemini is evaluating your answer..."
                ):

                    try:

                        clean_resume = (
                            clean_resume_for_ai(
                                st.session_state.resume_text
                            )
                        )

                        feedback = evaluate_answer(
                            selected_question,
                            answer,
                            clean_resume,
                            job_description
                        )

                        st.session_state.answer_feedback = (
                            feedback
                        )

                    except Exception as e:

                        st.error(
                            "Gemini could not evaluate "
                            "your answer."
                        )

                        st.code(str(e))

        # ----------------------------------------------------
        # Display feedback
        # ----------------------------------------------------

        if st.session_state.answer_feedback:

            st.divider()

            st.subheader(
                "📊 Interview Feedback"
            )

            st.markdown(
                st.session_state.answer_feedback
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HireLens helps students compare their resume "
    "with job descriptions and prepare for interviews. "
    "Results are AI-generated and should be used as guidance."
)