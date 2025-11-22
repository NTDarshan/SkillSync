from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
import json
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Resume Analyzer API", version="1.0.0")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Model initialization
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Define the state
class ResumeState(BaseModel):
    """State model for resume analysis workflow"""
    resume_text: str = Field(..., description="Raw extracted text from resume")
    job_description: str = Field(..., description="Job description text")
    
    # Contact Info
    candidate_name: Optional[str] = Field(None, description="Candidate's full name")
    candidate_email: Optional[str] = Field(None, description="Candidate's email address")
    
    # Skills
    resume_skills: List[str] = Field(default_factory=list, description="Skills extracted from resume")
    required_skills: List[str] = Field(default_factory=list, description="Skills required by job description")
    matching_skills: List[str] = Field(default_factory=list, description="Skills present in both resume and job requirements")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills missing from resume")
    
    # Experience
    candidate_experience_years: Optional[float] = Field(None, description="Total years of experience from resume")
    required_experience_years: Optional[float] = Field(None, description="Required years of experience")
    experience_summary: Optional[str] = Field(None, description="Summary of candidate's experience")
    
    # Final results
    final_score: Optional[float] = Field(None, description="Final computed score (0–100)")
    final_recommendation: Optional[Literal["hire", "maybe", "no"]] = Field(
        None, description="Final recommendation"
    )
    detailed_report: Optional[str] = Field(None, description="Detailed hiring recommendation report")
    email_template: Optional[str] = Field(None, description="Email template to send to candidate")
    
    class Config:
        # Allow mutation for LangGraph state updates
        validate_assignment = True


# Pydantic models for structured outputs
class ContactInfo(BaseModel):
    """Model for extracting contact information from resume"""
    name: str = Field(description="Candidate's full name")
    email: str = Field(description="Candidate's email address")


class SkillsList(BaseModel):
    """Model for extracting skills as a structured list"""
    skills: List[str] = Field(description="List of skills, technologies, tools, or certifications")


class ExperienceInfo(BaseModel):
    """Model for extracting experience information"""
    years_of_experience: float = Field(description="Total years of professional experience")
    summary: str = Field(description="Brief summary of work experience and roles")


class JobRequirements(BaseModel):
    """Model for extracting job requirements"""
    required_skills: List[str] = Field(description="Required skills and technologies")
    required_experience_years: float = Field(description="Minimum years of experience required")


class SkillsMatchResult(BaseModel):
    """Model for skill matching results"""
    matching: List[str] = Field(description="Skills present in both resume and job requirements, including similar/equivalent skills")
    missing: List[str] = Field(description="Required skills that are missing from the resume")


class ScoringResult(BaseModel):
    """Model for LLM-based scoring"""
    score: float = Field(description="Overall compatibility score from 0-100", ge=0, le=100)
    recommendation: Literal["hire", "maybe", "no"] = Field(description="Hiring recommendation")
    

class DetailedReport(BaseModel):
    """Model for detailed hiring recommendation report"""
    report: str = Field(description="Comprehensive, well-formatted hiring recommendation report")


class EmailTemplate(BaseModel):
    """Model for email template"""
    email_content: str = Field(description="Professional email template for candidate communication")


# Create structured output models
structured_contact_extractor = model.with_structured_output(ContactInfo)
structured_skills_extractor = model.with_structured_output(SkillsList)
structured_experience_extractor = model.with_structured_output(ExperienceInfo)
structured_job_requirements_extractor = model.with_structured_output(JobRequirements)
structured_skills_matcher = model.with_structured_output(SkillsMatchResult)
structured_scorer = model.with_structured_output(ScoringResult)
structured_report_generator = model.with_structured_output(DetailedReport)
structured_email_generator = model.with_structured_output(EmailTemplate)


# Define the node functions
def extract_contact_info(state: ResumeState) -> dict:
    """Extract candidate's contact information from resume"""
    resume_text = state.resume_text
    
    prompt = f"""You are an expert resume parser. Extract the candidate's contact information from the following resume.

Find:
- Full name (first name and last name)
- Email address

If you cannot find the email, use "not_provided@example.com" as a placeholder.

Resume:
{resume_text}

Extract the contact information:"""
    
    result = structured_contact_extractor.invoke(prompt)
    
    return {
        "candidate_name": result.name,
        "candidate_email": result.email
    }


def extract_resume_skills(state: ResumeState) -> dict:
    """Extract skills from the resume using structured output"""
    resume_text = state.resume_text
    
    prompt = f"""You are an expert resume analyst. Extract ALL skills, technologies, tools, frameworks, databases, programming languages, certifications, and competencies from the following resume.

Be comprehensive and include:
- Programming languages (e.g., Python, JavaScript, TypeScript, Go, C#)
- Frameworks and libraries (e.g., React, Node.js, Express.js, Next.js)
- Databases (e.g., MongoDB, MySQL, PostgreSQL)
- Cloud platforms (e.g., AWS, Google Cloud, Azure)
- Tools and platforms (e.g., Git, Docker, Kubernetes)
- Soft skills (e.g., Communication, Leadership)
- Certifications and methodologies

Resume:
{resume_text}

Extract all skills you can find:"""
    
    result = structured_skills_extractor.invoke(prompt)
    
    return {"resume_skills": result.skills}


def extract_resume_experience(state: ResumeState) -> dict:
    """Extract experience information from resume"""
    resume_text = state.resume_text
    
    prompt = f"""You are an expert resume analyst. Extract the candidate's work experience information from the following resume.

Analyze:
- Total years of professional experience (calculate from dates if available)
- Key roles and positions held
- Industries and domains worked in
- Notable projects or achievements

Resume:
{resume_text}

Provide the total years of experience and a brief summary:"""
    
    result = structured_experience_extractor.invoke(prompt)
    
    return {
        "candidate_experience_years": result.years_of_experience,
        "experience_summary": result.summary
    }


def extract_job_requirements(state: ResumeState) -> dict:
    """Extract all requirements from job description including skills and experience"""
    job_description = state.job_description
    
    prompt = f"""You are an expert job requirements analyst. Extract ALL requirements from this job description.

Extract:
1. Required skills, technologies, tools, frameworks, and competencies
2. Minimum years of experience required (if mentioned, otherwise estimate based on seniority level)

Be comprehensive with skills and include:
- Technical skills and programming languages
- Frameworks and libraries
- Databases and data technologies
- Cloud and DevOps tools
- Soft skills and competencies
- Required certifications or methodologies

Job Description:
{job_description}

Extract all requirements:"""
    
    result = structured_job_requirements_extractor.invoke(prompt)
    
    return {
        "required_skills": result.required_skills,
        "required_experience_years": result.required_experience_years
    }


def match_skills(state: ResumeState) -> dict:
    """Match skills between resume and job requirements using structured output"""
    resume_skills = state.resume_skills
    required_skills = state.required_skills
    
    prompt = f"""You are an expert skill matching analyst. Compare the candidate's skills with the job requirements.

IMPORTANT MATCHING RULES:
1. Match exact skills (e.g., "Python" in both lists)
2. Match equivalent/similar skills (e.g., "React.js" matches "React", "AWS EC2" matches "AWS")
3. Match related skills (e.g., "Node.js" and "Express.js" are related to backend development)
4. Be generous with matching - if skills are clearly related or one is a subset of another, count it as a match

Candidate's Skills:
{json.dumps(resume_skills, indent=2)}

Job Required Skills:
{json.dumps(required_skills, indent=2)}

Identify:
- Matching skills: Skills from the required list that the candidate has (exactly or equivalently)
- Missing skills: Required skills that the candidate clearly does not have

Provide your analysis:"""
    
    result = structured_skills_matcher.invoke(prompt)
    
    return {
        "matching_skills": result.matching,
        "missing_skills": result.missing
    }


def calculate_score(state: ResumeState) -> dict:
    """Calculate compatibility score using LLM, considering both skills and experience"""
    
    prompt = f"""You are an expert hiring analyst. Calculate a comprehensive compatibility score (0-100) for this candidate based on BOTH their skills match and experience level.

Candidate Profile:
- Resume Skills: {json.dumps(state.resume_skills)}
- Experience: {state.candidate_experience_years} years
- Experience Summary: {state.experience_summary}

Job Requirements:
- Required Skills: {json.dumps(state.required_skills)}
- Required Experience: {state.required_experience_years} years

Skill Analysis:
- Matching Skills ({len(state.matching_skills)}): {json.dumps(state.matching_skills)}
- Missing Skills ({len(state.missing_skills)}): {json.dumps(state.missing_skills)}

Scoring Guidelines:
1. Skills Match: 60% weight
   - Consider both quantity and quality of matching skills
   - Penalize critical missing skills more heavily
   
2. Experience Level: 40% weight
   - Compare candidate's experience with required experience
   - Consider if experience is relevant to the role
   - Overqualification is okay, underqualification should lower score

Recommendation Thresholds:
- 75-100: "hire" - Strong match, highly recommended
- 50-74: "maybe" - Partial match, could work with training
- 0-49: "no" - Weak match, significant gaps

Provide your score (0-100) and recommendation:"""
    
    result = structured_scorer.invoke(prompt)
    
    return {
        "final_score": result.score,
        "final_recommendation": result.recommendation
    }


def generate_detailed_report(state: ResumeState) -> dict:
    """Generate a comprehensive, formatted hiring recommendation report"""
    
    prompt = f"""You are a senior hiring manager. Generate a comprehensive, professional hiring recommendation report for this candidate.

Candidate Profile:
- Resume Skills: {json.dumps(state.resume_skills)}
- Experience: {state.candidate_experience_years} years
- Experience Summary: {state.experience_summary}

Job Requirements:
- Required Skills: {json.dumps(state.required_skills)}
- Required Experience: {state.required_experience_years} years

Analysis Results:
- Compatibility Score: {state.final_score}%
- Recommendation: {state.final_recommendation.upper()}
- Matching Skills ({len(state.matching_skills)}): {json.dumps(state.matching_skills)}
- Missing Skills ({len(state.missing_skills)}): {json.dumps(state.missing_skills)}

Generate a well-formatted report with the following sections:

1. **EXECUTIVE SUMMARY** (2-3 sentences)
   - Overall recommendation and key highlights

2. **SKILLS ASSESSMENT**
   - ✅ Strengths: List 3-5 key matching skills and why they're valuable
   - ❌ Gaps: List critical missing skills and their impact
   - Skills Match Rate: X out of Y required skills

3. **EXPERIENCE EVALUATION**
   - Years of experience comparison (candidate vs required)
   - Relevant experience highlights
   - How experience level impacts role fit

4. **REASONS TO HIRE** (if score >= 50)
   - List 3-5 compelling reasons why this candidate should be considered
   - Highlight unique strengths or transferable skills

5. **CONCERNS & RISKS** (if score < 75)
   - List 2-4 concerns or skill gaps
   - Mitigation strategies (training, mentorship, etc.)

6. **FINAL VERDICT**
   - Clear hiring recommendation with confidence level
   - Suggested next steps (interview, technical assessment, reject, etc.)

Make it professional, specific, and actionable. Use bullet points and clear formatting."""
    
    result = structured_report_generator.invoke(prompt)
    
    return {"detailed_report": result.report}


def generate_email_template(state: ResumeState) -> dict:
    """Generate personalized email template for candidate"""
    
    recommendation = state.final_recommendation
    
    if recommendation == "hire":
        prompt = f"""You are a hiring manager. Generate a professional, warm, and encouraging email to inform the candidate that they have been SELECTED for the position.

Candidate Information:
- Name: {state.candidate_name}
- Email: {state.candidate_email}
- Score: {state.final_score}%

Key Strengths:
- Matching Skills: {', '.join(state.matching_skills[:5])}...
- Experience: {state.candidate_experience_years} years

Email Requirements:
1. Start with a professional greeting using their name
2. Congratulate them on being selected
3. Highlight 2-3 key reasons why they were chosen (based on skills and experience)
4. Mention next steps (e.g., schedule interview, technical assessment)
5. Express enthusiasm about them joining the team
6. Professional closing

Keep it warm, professional, and encouraging. Use proper email format with subject line."""
    
    elif recommendation == "maybe":
        prompt = f"""You are a hiring manager. Generate a professional email to inform the candidate that they are being considered but we need additional evaluation.

Candidate Information:
- Name: {state.candidate_name}
- Email: {state.candidate_email}
- Score: {state.final_score}%

Analysis:
- Matching Skills: {', '.join(state.matching_skills[:3])}
- Missing Skills: {', '.join(state.missing_skills[:3])}
- Experience: {state.candidate_experience_years} years (Required: {state.required_experience_years})

Email Requirements:
1. Professional greeting using their name
2. Thank them for applying and acknowledge their strong points
3. Explain that we need additional evaluation (technical assessment, further interviews)
4. Mention specific areas they could strengthen
5. Provide next steps or timeline
6. Keep the tone positive and encouraging

Use proper email format with subject line."""
    
    else:  # recommendation == "no"
        prompt = f"""You are a hiring manager. Generate a professional, respectful, and constructive email to inform the candidate that they were NOT selected for this position.

Candidate Information:
- Name: {state.candidate_name}
- Email: {state.candidate_email}
- Score: {state.final_score}%

Analysis:
- Matching Skills: {', '.join(state.matching_skills) if state.matching_skills else 'Limited match'}
- Missing Critical Skills: {', '.join(state.missing_skills[:5])}...
- Experience: {state.candidate_experience_years} years (Required: {state.required_experience_years})

Email Requirements:
1. Professional greeting using their name
2. Thank them sincerely for their time and interest
3. Politely inform them they were not selected for THIS specific role
4. Provide 2-3 constructive reasons (be tactful but honest):
   - Mention skill gaps professionally
   - Suggest areas for improvement
5. Encourage them to apply for future opportunities if they develop the missing skills
6. Wish them success in their job search
7. Professional, respectful closing

Be kind, constructive, and professional. Use proper email format with subject line."""
    
    result = structured_email_generator.invoke(prompt)
    
    return {"email_template": result.email_content}


# Define the graph
graph = StateGraph(ResumeState)

# Add nodes
graph.add_node("extract_contact_info", extract_contact_info)
graph.add_node("extract_resume_skills", extract_resume_skills)
graph.add_node("extract_resume_experience", extract_resume_experience)
graph.add_node("extract_job_requirements", extract_job_requirements)
graph.add_node("match_skills", match_skills)
graph.add_node("calculate_score", calculate_score)
graph.add_node("generate_detailed_report", generate_detailed_report)
graph.add_node("generate_email_template", generate_email_template)

# Define the workflow edges
graph.add_edge(START, "extract_contact_info")
graph.add_edge("extract_contact_info", "extract_resume_skills")
graph.add_edge("extract_resume_skills", "extract_resume_experience")
graph.add_edge("extract_resume_experience", "extract_job_requirements")
graph.add_edge("extract_job_requirements", "match_skills")
graph.add_edge("match_skills", "calculate_score")
graph.add_edge("calculate_score", "generate_detailed_report")
graph.add_edge("generate_detailed_report", "generate_email_template")
graph.add_edge("generate_email_template", END)

# Compile the graph
workflow = graph.compile()


# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")


# Process resume function using LangGraph workflow
def process_resume(state: ResumeState) -> ResumeState:
    """
    Process the resume using LangGraph workflow.
    This orchestrates all analysis steps through the compiled graph.
    """
    # Convert Pydantic model to dict for workflow
    state_dict = state.model_dump()
    
    # Run the workflow
    result = workflow.invoke(state_dict)
    
    # Convert result back to ResumeState
    return ResumeState(**result)


# API Response model
class AnalysisResponse(BaseModel):
    """Response model for resume analysis"""
    candidate_name: str
    candidate_email: str
    resume_skills: List[str]
    required_skills: List[str]
    matching_skills: List[str]
    missing_skills: List[str]
    candidate_experience_years: float
    required_experience_years: float
    experience_summary: str
    final_score: float
    final_recommendation: Literal["hire", "maybe", "no"]
    detailed_report: str
    email_template: str


# Define the API endpoint
@app.post("/upload", response_model=AnalysisResponse)
async def upload_resume(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
):
    """
    Upload a resume PDF and job description for analysis.
    
    Returns:
    - Extracted skills from resume and job description
    - Matching and missing skills
    - Compatibility score (0-100)
    - Hiring recommendation (hire/maybe/no)
    """
    # Validate file type
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Extract text from the uploaded PDF resume
        resume_text = extract_text_from_pdf(resume.file)
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Create initial state object
        state = ResumeState(
            resume_text=resume_text,
            job_description=job_description,
        )
        
        # Process the resume through the workflow
        result_state = process_resume(state)
        
        # Return the comprehensive analysis response
        return AnalysisResponse(
            candidate_name=result_state.candidate_name,
            candidate_email=result_state.candidate_email,
            resume_skills=result_state.resume_skills,
            required_skills=result_state.required_skills,
            matching_skills=result_state.matching_skills,
            missing_skills=result_state.missing_skills,
            candidate_experience_years=result_state.candidate_experience_years,
            required_experience_years=result_state.required_experience_years,
            experience_summary=result_state.experience_summary,
            final_score=result_state.final_score,
            final_recommendation=result_state.final_recommendation,
            detailed_report=result_state.detailed_report,
            email_template=result_state.email_template
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Resume Analyzer API",
        "version": "1.0.0"
    }
   