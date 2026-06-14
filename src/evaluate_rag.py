from rag_pipeline import query_rag

questions = [
    "Who are the Morlocks and where do they live?",
    "Who are the Eloi?",
    "What happens to Weena?",
    "Why does the Time Traveller fear the Morlocks?",
    "What happens to the Time Machine?"
]

for i, q in enumerate(questions, start=1):

    print(f"\n{'='*80}")
    print(f"Question {i}: {q}")
    print(f"{'='*80}")

    answer_no_rag = query_rag(q, use_rag=False)

    answer_rag = query_rag(
        q,
        use_rag=True,
        top_k=8
    )

    print("\nWITHOUT RAG")
    print(answer_no_rag)

    print("\nWITH RAG")
    print(answer_rag)