# Agentic Workflow를 이용하여 복잡한 문제 해결하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fagentic-solver&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>



여기에서는 LangGraph를 이용하여 plan and execute 패턴의 agentic workflow를 구현하고, 이를 이용하여 복잡한 문제를 해결하는 방법에 대해 설명합니다. LLM의 한글 능력 및 지능적 분석 능력을 확인하기 위하여 대학 수학 능력 시험중에 국어 영역에 대해 테스트를 수행하였습니다.

## 수능 국어 문제 

### 복잡한 문제로 수능 국어를 선택한 이유

수학 능력 시험의 국어 영역은 LLM을 이용한 어플리케이션의 한국어에 대한 이해를 측정하기 좋은 주제입니다. [지문과 선택지-화법과 작문](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)은 json포맷으로 문제와 답을 제공하고 있습니다. 또한, [수능 문제의 경우에 정답이 알려져있고 상세한 해설서](https://m.blog.naver.com/awesome-2030/222931282476)도 결과를 확인할 때에 참고할 수 있습니다. 또한, [2023년 수능의 국어(화법과 작문)의 1등급 컷](https://www.nextplay.kr/news/articleView.html?idxno=4617)은 92점입니다. 

여기에서는 [Anthropic의 Claude Sonnet 3.5](https://www.anthropic.com/news/claude-3-5-sonnet)와 [LangGraph를 구현한 plan and execute 패턴](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute)의 agentic workflow를 이용하여 92점을 얻었습니다. 


## 전체 Architecture

여기에서 사용한 Architecture는 아래와 같습니다. API Gateway를 이용하여 클라이언트와 WebSocket으로 대화를 수행하고, AWS Lambda와 LangGraph를 이용하여 agentic workflow를 구현합니다. Workflow를 수행하기 위하여 외부 저장소나 인터넷 검색이 필요한 경우에는 Vector/Lexical 검색이 가능한 Knowledge Base와 Tavily 검색을 활용합니다. 다만, 수능 국어 문제의 경우에는 국어 자체에 대한 해석 능력을 확인하기 위하여 외부 데이터를 활용하지 않고 LLM의 지적능력만을 활용하였습니다. 여기에서는 plan and execute 패턴방식의 workflow를 사용하므로 multi region을 이용한 병렬처리를 통해 속도와 LLM의 throttling 이슈를 해결합니다. 

<img width="652" alt="image" src="https://github.com/user-attachments/assets/d873497d-b7ec-4043-9a1f-ebe37c1c2bcf">


### Agentic Workflow 구현하기

Plan and exeuction 패턴을 이용하면 복잡한 문제를 step by step으로 처리할 수 있습니다. 이때 LangGraph를 이용해 agentic workflow를 구현하는 것은 [LangGraph로 구현하는 Agentic Workflow](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute)을 참조합니다. Workflow의 노드들 간에 데이터 교환을 위해 State 클래스를 정의합니다. 

```python
class State(TypedDict):
    plan: list[str]
    past_steps: Annotated[List[Tuple], operator.add]
    info: Annotated[List[Tuple], operator.add]
    paragraph: str
    question: str
    question_plus: str
    choices: list[str]
    answer: str
    select: int
```


Agentic workflow를 아래와 같이 정의합니다. 여기에는 plan, execute, replan, final_answer로 노드를 생성하고 한개의 conditional edge인 should_end를 가지고 있습니다. 

```python
def buildPlanAndExecute():
    workflow = StateGraph(State)
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("replaner", replan_node)
    workflow.add_node("final_answer", final_answer)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replaner")
    workflow.add_conditional_edges(
        "replaner",
        should_end,
        {
            "continue": "executor",
            "end": "final_answer",
        },
    )
    workflow.add_edge("final_answer", END)

    return workflow.compile()
```

아래의 activity diagram을 이용하면 복잡한 workflow의 동작을 쉽게 이해할 수 있습니다. 

![image](https://github.com/user-attachments/assets/301dae63-30aa-4ebc-b434-fb942fa54e85)



수능 국어 문제에서는 지문인 paragraph가 주어지고 경우에 따라 보기가 주어지고, 보통 5개정도의 선택지가 주어집니다. 아래와 같이 step by step형태로 계획을 세울수 있도록 프롬프트를 준비합니다. 각 단계를 list로 관리하기 위하여 아래와 같이 한줄로 질문을 해결하는 단계를 생성하도록 예제를 이용해 프롬프트를 작성하였습니다. 

```python
def plan_node(state: State, config):
    print("###### plan ######")
            
    list_choices = ""
    choices = state["choices"]
    for i, choice in enumerate(choices):
        list_choices += f"({i+1}) {choice}\n"
    
    system = (
        "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."                
        
        "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
        "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
        "1. [질문을 해결하기 위한 단계]"
        "2. [질문을 해결하기 위한 단계]"
        "..."                
    )
    
    human = (
        "<paragraph> tag의 주어진 문장을 참조하여 <question> tag의 질문에 대한 적절한 답변을 <choice> tag안에서 선택하가 위한 단계별 계획을 세우세요."
        "단계별 계획에 <result> tag를 붙여주세요."
        
        "주어진 문장:"
        "<paragraph>"
        "{paragraph}"
        "</paragraph>"

        "질문:"
        "<question>"
        "{question}"
                        
        "{question_plus}"                
        "</question>"

        "선택지:"
        "<choices>"
        "{list_choices}"
        "</choices>"
    )
                        
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    chat, select = get_llm(state["select"])
    planner = planner_prompt | chat
    response = planner.invoke({
        "paragraph": paragraph,
        "question": question,
        "question_plus": question_plus,
        "list_choices": list_choices
    })
    result = response.content
    output = result[result.find('<result>')+8:result.find('</result>')]
    
    plan = output.strip().replace('\n\n', '\n')
    planning_steps = plan.split('\n')
    
    return {
        "plan": planning_steps,
        "select": select
    }
```

세워진 계획에서 첫번째 계획을 수행합니다. 여기서 주어진 문제를 paragraph, question, choice tag를 이용해 설명하고, 첫번째 계획을 task로 수행합니다. 이때 LLM이 충분히 생각하도록 모든 선택지에 대한 근거를 설명하도록 요청하고 선택지에서 한개를 고르도록 요청합니다. Agentic workflow는 결과가 나올때까지 반복하게 되어서 수행시간이 길어집니다. 수행시간을 단축하기 위하여 LLM이 현재의 선택에 대한 신뢰도(confidence)를 제시하도록 하고, 최고 신뢰도(여기서는 5)인 경우에 실행을 완료하도록 합니다. 신뢰도가 최고값을 가지면 계획(plan)을 비워서 다음 노드에서 완료되도록 합니다. 

```python
def execute_node(state: State, config):
    print("###### execute ######")
    plan = state["plan"]
    
    list_choices = ""
    choices = state["choices"]
    for i, choice in enumerate(choices):
        list_choices += f"({i+1}) {choice}\n"
    
    task = plan[0]
    context = ""
    for info in state['info']:
        if isinstance(info, HumanMessage):
            context += info.content+"\n"
        else:
            context += info.content+"\n\n"
                    
    system = (
        "당신은 국어 수능문제를 푸는 일타강사입니다."
    )
    human = (
        "당신의 목표는 <paragraph> tag의 주어진 문장으로 부터 <question> tag의 주어진 질문에 대한 적절한 답변을 <choice> tag의 선택지에서 찾는것입니다."
        "<previous_result> tag에 있는 이전 단계의 결과를 참조하여, <task> tag의 실행 단계를 수행하고 적절한 답변을 구합니다."
        "문제를 풀이할 때 모든 선택지마다 근거를 주어진 문장에서 찾아 설명하세요."
        "선택지의 주요 단어들의 의미를 주어진 문장과 비교해서 꼼꼼히 차이점을 찾습니다."
        "질문에 대한 답을 선택지 중에 한 개만 골라서 대답해야 합니다."
        "최종 결과의 번호에 <result> tag를 붙여주세요."
        "최종 결과의 신뢰도를 1-5 사이의 숫자로 나타냅니다. 신뢰되는 <confidence> tag를 붙입니다."  
                            
        "주어진 문장:"
        "<paragraph>"
        "{paragraph}"
        "</paragraph>"
            
        "주어진 질문:"
        "<question>"
        "{question}"
            
        "{question_plus}"
        "</question>"
        
        "선택지:"
        "<choices>"
        "{list_choices}"
        "</choices>"
        
        "이전 단계의 결과"
        "<previous_result>"
        "{info}"
        "</previous_result>"

        "실행 단계:"
        "<task>"
        "{task}"
        "</task>"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    chat, select = get_llm(state["select"])
    chain = prompt | chat                        
    response = chain.invoke({
        "paragraph": state["paragraph"],
        "question": state["question"],
        "question_plus": state["question_plus"],
        "list_choices": list_choices,
        "info": context,
        "task": task
    })
    
    result = response.content
    confidence = result[result.find('<confidence>')+12:result.find('</confidence>')]    
    transaction = [HumanMessage(content=task), AIMessage(content=result)]
    
    if confidence == "5":
        plan = []
        answer = result
    else:
        plan = state["plan"]
        answer = ""
    
    return {
        "plan": plan,
        "info": transaction,
        "past_steps": [task],
        "answer": answer,
        "select": select
    }
```

처음 생성한 계획을 이후 실행 과정에서 업데이트하면 더 좋은 결과를 얻을 있습니다. Execution 노드에서 첫번째 계획을 세웠으므로 replan 노드에서는 실행한 계획을 제외한 계획을 업데이트 합니다. 현재의 목표를 remind 시키고 나서, 원래의 계획과 완료된 계획을 알려주고 새로운 계획을 구성하도록 프롬프트를 구성합니다. 수정된 계획을 프롬프트를 이용해 한줄씩으로 정의하도록 하고 state의 plan을 업데이트 합니다.  

```python
def replan_node(state: State, config):
    print('#### replan ####')            
    list_choices = ""
    choices = state["choices"]
    for i, choice in enumerate(choices):
        list_choices += f"({i+1}) {choice}\n"
    
    if len(state["plan"])==0:
        return {"plan": []}
    
    system = (
        "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
    )        
    human = (
        "당신의 목표는 <paragraph> tag의 주어진 문장으로 부터 <question> tag의 주어진 질문에 대한 적절한 답변을 <choice> tag안에서 선택지에서 찾는것입니다."
        
        "주어진 문장:"
        "<paragraph>"
        "{paragraph}"
        "</paragraph>"
        
        "주어진 질문:"
        "<question>"
        "{question}"
        
        "{question_plus}"
        "</question>"
        
        "선택지:"
        "<list_choices>"
        "{list_choices}"
        "</list_choices>"
        
        "당신의 원래 계획은 아래와 같습니다." 
        "<original_plan>"                
        "{plan}"
        "</original_plan>"

        "완료한 단계는 아래와 같습니다."
        "<past_steps>"
        "{past_steps}"
        "</past_steps>"
        
        "당신은 <original_plan> tag의 원래 계획을 상황에 맞게 수정하세요."
        "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 단계는 계획에 포함하지 마세요."                
        "수정된 계획에는 <plan> tag를 붙여주세요."
        "만약 더 이상 계획을 세우지 않아도 <question> tag의 주어진 질문에 답변할 있다면, 최종 결과로 <question>에 대한 답변을 <result> tag를 붙여 전달합니다."
        
        "수정된 계획의 형식은 아래와 같습니다."
        "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
        "1. [질문을 해결하기 위한 단계]"
        "2. [질문을 해결하기 위한 단계]"
        "..."         
    )                    
    replanner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )        
    chat, select = get_llm(state["select"])
    replanner = replanner_prompt | chat        
    response = replanner.invoke({
        "paragraph": state["paragraph"],
        "question_plus": state["question_plus"],
        "question": state["question"],
        "list_choices": list_choices,
        "plan": state["plan"],
        "past_steps": state["past_steps"]
    })
    result = response.content
    
    if result.find('<plan>') == -1:
        return {"plan":[], "select":select, "answer":result}
    else:
        output = result[result.find('<plan>')+6:result.find('</plan>')]
        
        plans = output.strip().replace('\n\n', '\n')
        planning_steps = plans.split('\n')
    
        return {"plan": planning_steps, "select":select}
```

Conditional edge인 should_end에서는 plan을 보고 계속 실행할지 종료할지를 결정합니다. 

```python
def should_end(state: State) -> Literal["continue", "end"]:
    print('#### should_end ####')    
    plan = state["plan"]
    print('plan: ', plan)
    if len(plan)<=1:
        next = "end"
    else:
        next = "continue"
    
    return next
```


## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


## 실행결과


### 수능 국어 문제

채팅 메뉴에서 파일을 선택하여 업로드를 수행합니다. 여기에서는 테스트 계정의 quota 한계로 [원본 파일](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)을 6개의 파일로 나눠서 테스트를 수행하였습니다.

[2023_11_KICE_1.json](./contents/2023_11_KICE_1.json)을 다운로드 후에 채팅창 하단의 파일 업로드 버튼을 선택하여 파일을 업로드한 후에 결과를 확인합니다. 아래와 같이 전체 문제를 맞추었습니다. 

![image](https://github.com/user-attachments/assets/c6950a3c-6b38-4305-a752-f7ab81b1e1a6)


[2023_11_KICE_2.json](./contents/2023_11_KICE_2.json)에 대한 결과는 아래와 같습니다. 2문제에서 오답이 나왔습니다. 

![image](https://github.com/user-attachments/assets/20836228-5738-45e3-b354-c18782566495)

오답인 15번 문제의 경우에는 본문의 그래프에 대한 이해가 필요하지만 json 파일에는 그림 파일에 대한 정보를 제공하지 않았습니다. 따라서 판단 불가로 처리되어서 답을 구하지 못하였습니다.

![noname](https://github.com/user-attachments/assets/d1e0b6b6-de74-41b7-a813-16cef48873b6)

또한 17번 문제의 경우에 지문의 그림과 함께 보기의 그림도 같이 이해가 필요하나 이에 대한 정보가 없어서 실패한것으로 보여집니다.

![image](https://github.com/user-attachments/assets/fa02aadd-bcca-4c79-903d-fef19f87e670)


[2023_11_KICE_3.json](./contents/2023_11_KICE_3.json)에 대한 결과는 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/90d7da8e-1b47-437e-8669-7fe188a7a06c)

오답 문제를 보면 아래와 같습니다. 

![image](https://github.com/user-attachments/assets/48eba351-0a63-4297-bc3c-bdd191e45c85)

로그로 실패한 이유를 알아보면 아래와 같습니다. 선택지인 (1)과 (3)이 적절하지 않다고 판단하였는데, (1)이 더 부적절하다고 판단한것으로 보여집니다. 도산십이곡의 초야우생을 알고 있었다면 정답을 선택할 수 있었을것으로 보여집니다.

```text
<보기>의 내용을 바탕으로 (가)와 (나)의 특징을 다음과 같이 파악할 수 있습니다:
(가) 도산십이곡:
- 강호는 자연의 이치와 인간이 지향하는 이치가 일치된 이상적 공간
- 조화로운 자연과 합일하는 화자가 등장
(나) 지수정가:
- 강호는 자연에서 생활하면서 자연의 가치를 새롭게 발견할 수 있는 공간
- 자연의 구체적인 모습을 묘사하며 자연의 가치를 확인한 화자가 등장
이러한 특징을 바탕으로 각 선택지를 분석하겠습니다.
<task>2. 각 선택지를 분석하여 적절성을 판단합니다.</task>
(1) "초야우생(草野愚生)이 이렇다 어떠하료"라는 표현은 화자가 자신을 겸손하게 표현한 것입니다. 그러나 이것이 반드시 "인간이 지향하는 이치와 자연의 이치가 일치된 공간"에 존재한다고 볼 수는 없습니다. 따라서 이 선택지는 적절하지 않습니다.
(2) "일조(一朝)에 진면목을 내 혼자 알았노라"는 표현은 화자가 자연에서 생활하면서 자연의 가치를 새롭게 발견했음을 나타냅니다. 이는 <보기>의 내용과 일치하므로 적절합니다.
(3) "천석고황(泉石膏肓)"은 자연을 사랑하는 마음이 고질병처럼 깊다는 의미로, 화자가 자연과 합일된 상태를 나타냅니다. 따라서 이를 아쉬움으로 해석하는 것은 적절하지 않습니다. 반면 "무릉도원"은 화자가 현실의 공간을 이상적 공간으로 인식하는 것을 보여주어 적절합니다. 이 선택지는 부분적으로만 적절합니다.
(4) "사시 가흥(佳興)이 사람과 한가지라"는 자연의 이치와 인간이 지향하는 이치가 일치함을 나타내며, "가지가지 다 좋구나"는 자연의 가치를 확인한 화자의 심정을 드러냅니다. 둘 다 <보기>의 내용과 일치하므로 적절합니다.
(5) "춘풍에 화만산하고 추야에 월만대라"는 조화로운 자연을, "벽류는 콸콸 흘러 옥 술잔을 때리는 듯"은 화자가 발견한 자연의 아름다움을 표현합니다. 둘 다 <보기>의 내용과 일치하므로 적절합니다.
<task>3. 분석 결과를 바탕으로 가장 적절하지 않은 선택지를 선택합니다.</task>
분석 결과, (1)번 선택지가 가장 적절하지 않습니다. '초야우생'이라는 표현이 반드시 이상적 공간에 존재하는 화자를 나타낸다고 볼 수 없으며, 이는 <보기>의 내용과도 직접적인 연관성이 없기 때문입니다.
```

[2023_11_KICE_4.json](./contents/2023_11_KICE_4.json)에 대한 결과는 아래와 같습니다. 모든 문제를 맞추었습니다. 

![image](https://github.com/user-attachments/assets/43d62f6d-5482-43e5-acbf-ca72efdbfb9b)


[2023_11_KICE_5.json](./contents/2023_11_KICE_5.json)에 대한 결과는 아래와 같습니다. 모두 정답을 맞추었습니다. 

![image](https://github.com/user-attachments/assets/d1513b21-e95b-46c3-bcb7-d1314eab2206)


[2023_11_KICE_6.json](./contents/2023_11_KICE_6.json)에 대한 결과는 아래와 같습니다. 모두 정답입니다. 

![image](https://github.com/user-attachments/assets/46471134-08ca-4b70-9cbc-92a27333ac0e)

이와 같이 전체 100점중에 92점을 획득하였습니다. 문제 점수에는 LLM의 지적능력이 매우 중요하며 프롬프트를 이용해 LLM이 문제에 대해 충분히 생각하도록 유도하여야 합니다. 

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

여기에서는 LangGraph를 이용하여 agentic workflow 방식의 agent를 구현하였습니다. 이를 통해 복잡한 문제를 plan and execute 패턴으로 해결하였습니다. 이 패턴은 CoT (Chain of Thought)처럼 문제를 step by step으로 풀도록 유도함으로써 복잡한 문제의 의미를 파악하여 더 좋은 결과를 얻을 수 있습니다. 이때 반복적인 동작이 필수적으로 요청되므로 zero shot 방식에 비하여 수행시간이 증가합니다. 여기에서는 multi region을 이용한 병렬처리를 통해 수행시간을 단축하고, 단계를 실행중에 신뢰도(confidence)확보하면 중단하는 방식으로 속도를 개선하였습니다. 오답은 문제를 이해할때 텍스트뿐 아니라 이미지를 같이 활용하거나, 더 지능적인 LLM으로 더 복잡한 문장 구조를 이해하는 방식으로 개선이 가능할 것으로 보여집니다. 

## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "api-agentic-solver", "api-chatbot-for-agentic-solver"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/environment/agentic-solver/cdk-agentic-solver/ && cdk destroy --all
```
