# Agentic Workflow를 이용하여 복잡한 문제 해결하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fagentic-solver&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>



여기에서는 LangGraph를 이용하여 plan and execute 패턴의 agentic workflow를 구현하고, 이를 이용하여 복잡한 문제를 해결하는 방법에 대해 설명합니다. 

## 수능 국어 문제 

### 복잡한 문제로 수능 국어를 선택한 이유

수학능력시험의 국어는 한국어에 대한 이해를 측정하기 위해 [지문과 선택지](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)가 주어집니다. 따라서, LLM의 언어 및 분석 능력을 판단하기에 매우 좋은 예제입니다. 또한, [수능 문제의 경우에 정답이 알려져있고 상세한 해설서](https://m.blog.naver.com/awesome-2030/222931282476)도 있습니다. 


## 전체 Architecture

여기에서 사용한 Architecture는 아래와 같습니다. API Gateway를 이용하여 클라이언트와 WebSocket으로 대화를 수행하고, AWS Lambda와 LangGraph를 이용하여 agentic workflow를 구현합니다. Workflow를 수행하기 위하여 외부 저장소나 인터넷 검색이 필요한 경우에는 Vector/Lexical 검색이 가능한 Knowledge Base와 Tavily 검색을 활용합니다. 다만, 수능 국어 문제의 경우에는 국어 자체에 대한 해석 능력을 확인하기 위하여 외부 데이터를 활용하지 않고 LLM의 지적능력만을 활용하였습니다. 여기에서는 plan and execute 패턴방식의 workflow를 사용하므로 multi region을 이용한 병렬처리를 통해 속도와 LLM의 throttling 이슈를 해결합니다. 

<img width="652" alt="image" src="https://github.com/user-attachments/assets/d873497d-b7ec-4043-9a1f-ebe37c1c2bcf">

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



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


## 실행결과


### 수능 국어 문제

채팅 메뉴에서 파일을 선택하여 업로드를 수행합니다. 여기에서는 테스트 계정의 quota 이슈로 5개의 파일로 나눠서 테스트를 수행하였습니다.

[2023_11_KICE_1.json](./contents/2023_11_KICE_1.json)을 다운로드 후에 채팅창 하단의 파일 업로드 버튼을 선택하여 파일을 업로드한 후에 결과를 확인합니다. 

![image](https://github.com/user-attachments/assets/c6950a3c-6b38-4305-a752-f7ab81b1e1a6)


#### 실패 문제의 분석

[2023_11_KICE_2.json](./contents/2023_11_KICE_2.json)에 대한 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/20836228-5738-45e3-b354-c18782566495)

15번 문제의 경우에는 본문의 그래프에 대한 이해가 필요하지만 json 파일에는 그림 파일에 대한 정보를 제공하지 않았습니다. 따라서 판단 불가로 처리되어서 답을 구하지 못하였습니다.

![noname](https://github.com/user-attachments/assets/d1e0b6b6-de74-41b7-a813-16cef48873b6)

또한 17번 문제의 경우에 지문의 그림과 함께 보기의 그림도 같이 이해가 필요하나 이에 대한 정보가 없어서 실패한것으로 보여집니다.

![image](https://github.com/user-attachments/assets/fa02aadd-bcca-4c79-903d-fef19f87e670)



### 수능 한국사 문제

여기에서 [한국사 영역의 시험문제](https://legendstudy.com/1574)의 한국사 문제를 참조하였습니다. 한국사의 경우는 인터넷 검색을 통해 지문에 대한 정보를 조회하고 이를 이용해 답변을 수행합니다. 한국사에 대한 자료를 RAG를 구현하면 더 정확한 검색 결과를 얻을 수 있지만 여기에서는 plan and execute 방식의 agentic workflow 패턴을 이용해 복잡한 workflow를 해결하는 방법에 대해 동작을 테스트 하기 위하여 인터넷 검색만을 활용하였습니다.

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

LangSmith를 보면 아래와 같이 1회 replan후 결과를 얻었습니다. 

![image](https://github.com/user-attachments/assets/1ff7e107-ae54-48c3-9401-a6f2b6fb1e71)

이때의 실행 단계는 아래와 같습니다.

```java
{
  "steps": [
    "대화의 핵심 내용 파악",
    "대화의 배경이 될 수 있는 역사적 사건 검토",
    "제시된 선택지 분석",
    "역사적 배경 확인",
    "최종 답안 선택"
  ]
}
```

### 메뉴의 Problem Solver 실행 결과

메뉴의 Problem Solver는 인터넷 검색 결과를 이용해 얻어진 정보를 활용합니다. 채팅창에서 "서울에서 부산을 거쳐서 제주로 가는 가장 저렴한 방법은?"이라는 질문을 하고 답변을 확인합니다. 

![noname](https://github.com/user-attachments/assets/dd49ee70-f086-4bf5-b9aa-dfb52ba3807f)

이때의 동작을 LangSmith로 확인합니다.

![image](https://github.com/user-attachments/assets/003efb54-d02a-414c-837a-207491cf4007)




## 결론

여기에서는 LangGraph를 이용하여 workflow 방식의 agent를 구현하였습니다. 이를 통해 복잡한 문제를 plan and execute 방식으로 해결하였습니다. 

## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "api-agentic-solver", "api-chatbot-for-agentic-solver"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/environment/agentic-solver/cdk-agentic-solver/ && cdk destroy --all
```
