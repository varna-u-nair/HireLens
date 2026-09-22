import re

import streamlit as st
from pypdf import PdfReader
from google import genai


# ============================================================
# GEMINI FUNCTIONS
# ============================================================

def generate_ai_questions(resume_text, job_description):
    client = genai.Client()

    prompt = f"""
You are an interview preparation assistant.

Create 5 interview questions for a student applying for the job below.

Use BOTH:
1. The student's resume
2. The job description

The questions should test:
- Technical skills required by the job
- Skills mentioned in the resume
- The student's projects or experience
- One question about a skill the student is missing

Job Description:
{job_description}

Resume:
{resume_text}

Return only the 5 questions as a numbered list.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
    except Exception:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

    return response.text


def evaluate_answer(question, answer, resume_text, job_description):
    client = genai.Client()

    prompt = f"""
You are an interview coach helping a student prepare for a job interview.

Evaluate the student's answer to the interview question.

Interview Question:
{question}

Student's Answer:
{answer}

Job Description:
{job_description}

Resume:
{resume_text}

Give feedback in exactly these sections:

1. What you did well
2. What could be improved
3. Important points you missed
4. A stronger example answer

Be constructive and specific.
Do not claim that your evaluation is an actual hiring decision.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
    except Exception:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

    return response.text


def generate_skill_recommendations(missing_skills, job_description):
    client = genai.Client()

    skills_text = ", ".join(missing_skills)

    prompt = f"""
You are a career guidance assistant.

The student is applying for this job:

Job Description:
{job_description}

The following skills were detected as missing from the student's resume:
{skills_text}

For each missing skill:
1. Explain briefly why it matters for this job.
2. Give 2-3 specific things the student should learn or practice.
3. Keep the advice practical for a student.

Return the result as a clear numbered list.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
    except Exception:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

    return response.text


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def contains_skill(text, skill):
    """
    Check whether a skill appears as a real word/phrase
    instead of just appearing as part of another word.
    """
    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


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

    # Remove common profile URLs
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

skills = [
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "html",
    "css",
    "sql",
    "pandas",
    "numpy",
    "machine learning",
    "deep learning",
    "data science",
    "git",
    "github",
    "streamlit",
    "django",
    "flask",
    "react",
    "tensorflow",
    "pytorch",
    "excel"
]


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "required_skills" not in st.session_state:
    st.session_state.required_skills = []

if "matched_skills" not in st.session_state:
    st.session_state.matched_skills = []

if "missing_skills" not in st.session_state:
    st.session_state.missing_skills = []

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "match_score" not in st.session_state:
    st.session_state.match_score = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "answer_feedback" not in st.session_state:
    st.session_state.answer_feedback = ""


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
    placeholder="Paste the job description here..."
)


# ============================================================
# ANALYZE
# ============================================================

if st.button("🚀 Analyze", use_container_width=True):

    if resume is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please paste a job description.")

    else:

        # ----------------------------------------------------
        # Read resume PDF
        # ----------------------------------------------------

        reader = PdfReader(resume)

        resume_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                resume_text += text + "\n"

        # Convert to lowercase for skill matching
        resume_text_lower = resume_text.lower()
        job_text = job_description.lower()

        # ----------------------------------------------------
        # Detect skills
        # ----------------------------------------------------

        resume_skills = []
        required_skills = []

        for skill in skills:

            if contains_skill(resume_text_lower, skill):
                resume_skills.append(skill)

            if contains_skill(job_text, skill):
                required_skills.append(skill)

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

        if len(required_skills) > 0:

            match_score = (
                len(matched_skills)
                / len(required_skills)
            ) * 100

        else:

            match_score = 0

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        st.session_state.analysis_done = True

        st.session_state.required_skills = required_skills

        st.session_state.matched_skills = matched_skills

        st.session_state.missing_skills = missing_skills

        st.session_state.match_score = match_score

        st.session_state.resume_text = resume_text

        # Reset previous AI results
        st.session_state.pop("ai_questions", None)
        st.session_state.pop("skill_recommendations", None)

        st.session_state.current_question = 0
        st.session_state.answer_feedback = ""


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done:

    st.divider()

    st.header("🎯 Your Match Score")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Job Match",
            f"{st.session_state.match_score:.0f}%"
        )

    with col2:

        st.metric(
            "Matched Skills",
            len(st.session_state.matched_skills)
        )

    with col3:

        st.metric(
            "Skills to Improve",
            len(st.session_state.missing_skills)
        )

    st.progress(
        int(st.session_state.match_score),
        text=(
            f"Overall Match: "
            f"{st.session_state.match_score:.0f}%"
        )
    )

    # ========================================================
    # MATCHED / MISSING SKILLS
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Matched Skills")

        if st.session_state.matched_skills:

            for skill in st.session_state.matched_skills:

                st.success(skill.title())

        else:

            st.info("No matching skills found.")

    with col2:

        st.subheader("❌ Missing Skills")

        if st.session_state.missing_skills:

            for skill in st.session_state.missing_skills:

                st.error(skill.title())

        else:

            st.success("No missing skills! 🎉")

    # ========================================================
    # RECOMMENDED SKILLS
    # ========================================================

    st.divider()

    st.header("📚 Recommended Skills")

    if st.session_state.missing_skills:

        st.write(
            "Skills you may want to improve:"
        )

        for skill in st.session_state.missing_skills:

            st.write(
                f"👉 **{skill.title()}**"
            )

    else:

        st.success(
            "Your resume covers all detected skills!"
        )

    # --------------------------------------------------------
    # AI SKILL GAP GUIDANCE
    # --------------------------------------------------------

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
                        "Something went wrong while generating "
                        "skill recommendations."
                    )

                    st.code(str(e))

    if "skill_recommendations" in st.session_state:

        st.divider()

        st.subheader("🧭 AI Skill Gap Guidance")

        st.write(
            st.session_state.skill_recommendations
        )

    # ========================================================
    # AI INTERVIEW MODE
    # ========================================================

    st.divider()

    st.header("🎤 AI Interview Mode")

    st.write(
        "Generate personalized interview questions using "
        "your resume and the job description."
    )

    if st.button(
        "🤖 Generate AI Questions",
        use_container_width=True
    ):

        # Remove unnecessary personal information
        # before sending resume text to Gemini.
        ai_resume_text = clean_resume_for_ai(
            st.session_state.resume_text
        )

        with st.spinner(
            "Gemini is creating your interview questions..."
        ):

            try:

                ai_questions = generate_ai_questions(
                    ai_resume_text,
                    job_description
                )

                st.session_state.ai_questions = ai_questions

                # Start from question 1
                st.session_state.current_question = 0
                st.session_state.answer_feedback = ""

            except Exception as e:

                st.error(
                    "Something went wrong while generating "
                    "the interview questions."
                )

                st.code(str(e))

    # ========================================================
    # INTERVIEW PRACTICE
    # ========================================================

    if "ai_questions" in st.session_state:

        questions = []

        for line in (
            st.session_state.ai_questions.splitlines()
        ):

            line = line.strip()

            # Accept formats such as:
            # 1. Question
            # 1) Question
            if (
                len(line) > 2
                and line[0].isdigit()
                and line[1] in ".)"
            ):

                questions.append(
                    line[2:].strip()
                )

        if questions:

            # Make sure the question number is valid
            if (
                st.session_state.current_question
                >= len(questions)
            ):

                st.session_state.current_question = 0

            current = (
                st.session_state.current_question
            )

            st.subheader(
                "🧠 AI Interview Practice"
            )

            st.write(
                f"Question {current + 1} "
                f"of {len(questions)}"
            )

            st.info(
                questions[current]
            )

            answer = st.text_area(
                "Your answer",
                key=f"answer_{current}",
                height=180,
                placeholder=(
                    "Type your interview answer here..."
                )
            )

            # ------------------------------------------------
            # Evaluate answer
            # ------------------------------------------------

            if st.button(
                "🤖 Evaluate My Answer",
                use_container_width=True
            ):

                if not answer.strip():

                    st.warning(
                        "Please write an answer first."
                    )

                else:

                    ai_resume_text = (
                        clean_resume_for_ai(
                            st.session_state.resume_text
                        )
                    )

                    with st.spinner(
                        "Gemini is evaluating your answer..."
                    ):

                        try:

                            feedback = evaluate_answer(
                                questions[current],
                                answer,
                                ai_resume_text,
                                job_description
                            )

                            st.session_state.answer_feedback = (
                                feedback
                            )

                        except Exception as e:

                            st.error(
                                "Something went wrong while "
                                "evaluating your answer."
                            )

                            st.code(str(e))

            # ------------------------------------------------
            # Feedback
            # ------------------------------------------------

            if st.session_state.answer_feedback:

                st.divider()

                st.subheader("📝 AI Feedback")

                st.write(
                    st.session_state.answer_feedback
                )

                # --------------------------------------------
                # Next question
                # --------------------------------------------

                if current < len(questions) - 1:

                    if st.button(
                        "➡️ Next Question",
                        use_container_width=True
                    ):

                        st.session_state.current_question += 1

                        st.session_state.answer_feedback = ""

                        st.rerun()

                else:

                    st.success(
                        "🎉 Interview practice complete!"
                    )

        else:

            st.warning(
                "Could not read the generated questions."
            )