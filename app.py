# app.py

import ollama

from tools import (
    tools,
    search_company_documents
)


# ---------------------------------------------------------
# 1. Ask the user a question
# ---------------------------------------------------------

question = input("You: ")


# ---------------------------------------------------------
# 2. Start the conversation
# ---------------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are a company policy assistant. "
            "Use the retrieved company documents to answer questions. "
            "Do not use your general knowledge to fill in missing information. "
            "If the retrieved documents do not contain enough information "
            "to answer the question, clearly say that the information "
            "is not available in the company documents."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# ---------------------------------------------------------
# 3. Agent loop
# ---------------------------------------------------------

while True:

    # Ask Qwen what to do.
    response = ollama.chat(
        model="qwen3:4b",
        messages=messages,
        tools=tools
    )

    assistant_message = response["message"]


    # -----------------------------------------------------
    # 4. Check whether Qwen requested a tool
    # -----------------------------------------------------

    tool_calls = assistant_message.get(
        "tool_calls",
        []
    )


    if not tool_calls:

        # No tool was requested.
        # Qwen is giving us the final answer.
        print("\nAssistant:")
        print(assistant_message["content"])

        break


    # -----------------------------------------------------
    # 5. Add Qwen's tool request to the conversation
    # -----------------------------------------------------

    messages.append(assistant_message)


    # -----------------------------------------------------
    # 6. Execute the requested tool
    # -----------------------------------------------------

    for tool_call in tool_calls:

        function_name = tool_call["function"]["name"]

        arguments = tool_call["function"]["arguments"]


        # Show which function the agent decided to call.
        print(
            f"\n[Agent called function: {function_name}]"
        )

        print(
            f"[Arguments: {arguments}]"
        )


        if function_name == "search_company_documents":

            question_for_search = arguments["question"]

            result = search_company_documents(
                question_for_search
            )


            # -------------------------------------------------
            # 7. Show that the function returned a result
            # -------------------------------------------------

            print("[Function returned a result]")


            # -------------------------------------------------
            # 8. Send the result back to Qwen
            # -------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": result
                }
            )