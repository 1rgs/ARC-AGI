import openai
#import instructor
import functools

# Define the model name


GPT4_MODEL_NAME = "gpt-4-turbo"
GPT4O_MODEL_NAME = "gpt-4o"
GPT4_VISION_MODEL_NAME = "gpt-4-turbo"
GPT3POINT5_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_MODEL = GPT4O_MODEL_NAME

def call_client(
    messages: list[dict],
    model_name: str=DEFAULT_MODEL,
    system_prompt="",
    response_model=None,
    organization="org-2DaYkiaLOqsFSSiS8ZdgUTmG",
    **kwargs
) -> str:
    if response_model:
        kwargs["response_model"] = response_model
    if "claude" in model_name:
        print("Calling Anthropic")
        anthropic_client = anthropic.Client(
            api_key=ANTHROPIC_AI_KEY,
        )
        return (
            anthropic_client.messages.create(
                system=system_prompt,
                model=model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            .content[0]
            .text
        )
    else:
        print("Calling OpenAI")
        #openai_client = instructor.patch(
        #    openai.OpenAI(api_key=OPENAI_API_KEY)
        #)
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        messages = [{"role": "system", "content": system_prompt}] + messages
        if response_model:
            return openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
                **kwargs,
            )
        else:
            print("n", kwargs.get("n", 1))
            if kwargs.get("n", 1) > 1:
                result = (
                    openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8,
                        **kwargs,
                    )
                )
                return [result.choices[i].message.content for i in range(kwargs.get("n", 1))]
            else:
                return (
                    openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8,
                        **kwargs,
                    )
                    .choices[0]
                    .message.content
                )

if __name__ == "__main__":
    res = call_client(messages = [{
        "role": "user", 
        "content": "What is the capital of France?"
    }])
    print(res)