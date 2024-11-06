# Agentic Workflow를 이용한 문제 해석기

여기에서는 agentic workflow를 이용하여 복잡한 문제를 해결하는 방법에 대해 설명합니다.


### Replan

Replan node는 아래와 같이 state를 가지고 새로운 plan을 생성합니다. state에는 이전 단계 응답 전체가 있습니다.

```python
def replan_node(state: State, config):
    update_state_message("replanning...", config)
    
    replanner_prompt = ChatPromptTemplate.from_template(
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer."
        "Do not add any superfluous steps."
        "The result of the final step should be the final answer."
        "Make sure that each step has all the information needed - do not skip steps."

        "Your objective was this:"
        "{input}"

        "Your original plan was this:"
        "{plan}"

        "You have currently done the follow steps:"
        "{past_steps}"

        "Update your plan accordingly."
        "If no more steps are needed and you can return to the user, then respond with that."
        "Otherwise, fill out the plan."
        "Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."
    )
    
    chat = get_chat()
    replanner = replanner_prompt | chat
    
    output = replanner.invoke(state)
    
    result = None
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(Act, include_raw=True)    
        info = structured_llm.invoke(output.content)
        print(f'attempt: {attempt}, info: {info}')
        
        if not info['parsed'] == None:
            result = info['parsed']
            break
                
    if result == None:
        return {"response": "답을 찾지 못하였습니다. 다시 시도해주세요."}
    else:
        if isinstance(result.action, Response):  # "parsed":"Act(action=Response(response="
            return {
                "response": result.action.response,
                "info": [result.action.response]
            }
        else:  # "parsed":"Act(action=Plan(steps=
            return {"plan": result.action.steps}    
```

이때, [flow-logs.md](https://github.com/kyopark2014/agentic-solver/blob/main/flow-logs.md)와 같이 replan의 결과로 Response 또는 Plan을 받게 됩니다.


## 실행 결과

"서울에서 부산을 거쳐서 제주로 가는 가장 저렴한 방법은?"의 질문에 대한 답변은 아래와 같습니다. 

![noname](https://github.com/user-attachments/assets/dd49ee70-f086-4bf5-b9aa-dfb52ba3807f)

이때의 동작을 LangSmith로 확인합니다.

![image](https://github.com/user-attachments/assets/003efb54-d02a-414c-837a-207491cf4007)
