"""LLM-as-judge grounding evaluator."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag import config
from rag.qa import Answer, format_context


class GroundingVerdict(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is supported by the context.")
    score: float = Field(ge=0.0, le=1.0, description="Fraction of the answer's claims supported by the context (0-1).")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims not supported by the context.")
    reasoning: str = Field(description="Brief justification.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator of retrieval-augmented answers. Judge ONLY whether "
            "the ANSWER is supported by the CONTEXT. Do not use outside knowledge and do not "
            "judge whether the answer is true in the real world. Break the answer into atomic "
            "claims, check each against the context, and list any claim that is not directly "
            "supported. Citation markers like [1] are not claims.",
        ),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"),
    ]
)


def get_judge(model: str | None = None) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    config.require_api_key()
    return ChatGoogleGenerativeAI(model=model or config.EVAL_MODEL, temperature=0)


def evaluate_grounding(
    question: str,
    answer: str,
    contexts: list[Document],
    judge: BaseChatModel | None = None,
) -> GroundingVerdict:
    if answer.strip() == config.FALLBACK_RESPONSE:
        return GroundingVerdict(
            grounded=True,
            score=1.0,
            reasoning="Fallback response: the system declined to answer, so no claims were made.",
        )
    chain = JUDGE_PROMPT | (judge or get_judge()).with_structured_output(GroundingVerdict)
    return chain.invoke(
        {"context": format_context(contexts), "question": question, "answer": answer}
    )


def evaluate_answer(ans: Answer, judge: BaseChatModel | None = None) -> GroundingVerdict:
    return evaluate_grounding(ans.question, ans.answer, ans.contexts, judge=judge)
