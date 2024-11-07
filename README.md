# Agentic Workflow를 이용하여 복잡한 문제 해결하기

여기에서는 agentic workflow를 이용하여 복잡한 문제를 해결하는 방법에 대해 설명합니다. 여기에서 사용하는 agentic workflow은 plan and exeuction입니다.

## Plan and Exeuction pattern

plan and exeuction pattern을 이용하면 복잡한 문제를 step by step으로 처리할 수 있습니다.

![image](https://github.com/user-attachments/assets/301dae63-30aa-4ebc-b434-fb942fa54e85)


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


#### 2023년 한국사 문제

[한국사 영역의 시험문제](https://legendstudy.com/1574)입니다.

```text
11. 다음 A, B 대화의 배경으로 가장 적절한 것은? 

A: 이보게,  종로에서 거의 날마다 보안회가 주관하는 대중 집회가 열리고 있다고 하네. 수천 명이 모여 한 뺌의 국토도 외국인에게 내줄 수 없다는 주장을 펼친다는 군
B. 지방에서는 이러한 주장에 호응하여 이곳 조곳에서 보안회에 의연금을 보낸다고합니다. 서울의 상인들도 가게 문을 닫고 이들의 투쟁을 지원한다더군요.

① 산미 증식 계획이 시행되었다. 
② 암태도 소작 쟁의가 발생하였다. 
③ 일본이 한국에 황무지 개간권을 요구하였다. 
④ 조선 총독부가 토지 조사 사업을 실시하였다. 
⑤ 회사 설립을 허가제로 하는 회사령이 제정되었다.
```

Agent의 경우에 인터넷 검색을 통해 아래와 같이 5번을 답했으나, 정답은 3번입니다.

![noname](https://github.com/user-attachments/assets/4d77bc30-ff66-4b25-b9b2-d6a6da2c42cd)

이때, plan and execute 패턴을 따르는 agentic solver는 아래와 같이 3번 정답을 찾을 수 있었습니다.

![noname](https://github.com/user-attachments/assets/3d05c25a-7d44-42a1-be7f-0f2fb78683a2)

