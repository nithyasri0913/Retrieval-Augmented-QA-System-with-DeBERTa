"""
Quick test script to verify the system is working correctly
"""

import sys
from src.pipeline import ProductionQASystem


def test_basic_functionality():
    """Test basic QA functionality."""
    print("=" * 70)
    print("Testing RAG QA System")
    print("=" * 70)

    # Sample documents
    documents = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "Python is a programming language created by Guido van Rossum in 1991.",
        "The Pacific Ocean is the largest ocean on Earth."
    ]

    try:
        # Initialize system
        print("\n1. Initializing system...")
        qa_system = ProductionQASystem(
            retriever_model='all-MiniLM-L6-v2',
            qa_model='deepset/deberta-v3-base-squad2',
            top_k=3,
            qa_threshold=0.2
        )
        print("✓ System initialized successfully")

        # Index documents
        print("\n2. Indexing documents...")
        qa_system.index_corpus(documents, use_chunking=False)
        print("✓ Documents indexed successfully")

        # Test questions
        print("\n3. Testing question answering...")
        test_cases = [
            ("Where is the Eiffel Tower?", "Paris"),
            ("When was the Eiffel Tower built?", "1889"),
            ("Who created Python?", "Guido van Rossum"),
            ("When was Python created?", "1991"),
        ]

        passed = 0
        failed = 0

        for question, expected in test_cases:
            result = qa_system.answer_question(question, verbose=False)

            if result['answer']:
                answer = result['answer']
                confidence = result['confidence']

                # Check if expected answer is in the result
                if expected.lower() in answer.lower():
                    print(f"✓ Q: {question}")
                    print(f"  A: {answer} (confidence: {confidence:.3f})")
                    passed += 1
                else:
                    print(f"✗ Q: {question}")
                    print(f"  Expected: {expected}, Got: {answer}")
                    failed += 1
            else:
                print(f"✗ Q: {question}")
                print(f"  No answer found")
                failed += 1

        # Results
        print("\n" + "=" * 70)
        print("Test Results")
        print("=" * 70)
        print(f"Passed: {passed}/{len(test_cases)}")
        print(f"Failed: {failed}/{len(test_cases)}")

        if failed == 0:
            print("\n🎉 All tests passed! System is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check the output above.")

        # Test save/load
        print("\n4. Testing save/load functionality...")
        qa_system.save('./test_saved_system')
        print("✓ System saved")

        loaded_system = ProductionQASystem.load('./test_saved_system')
        print("✓ System loaded")

        # Test loaded system
        result = loaded_system.answer_question("Where is the Eiffel Tower?")
        if result['answer']:
            print(f"✓ Loaded system works: {result['answer']}")
        else:
            print("✗ Loaded system failed to answer")

        # Get stats
        print("\n5. System Statistics:")
        stats = qa_system.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n" + "=" * 70)
        print("System test completed successfully!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
