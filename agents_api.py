from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
load_dotenv()
import os
from datetime import date

google_api_key=os.getenv("GOOGLE_API_KEY")

llm=LLM(model="gemini/gemini-1.5-pro", temperature=1, api_key=google_api_key)

app = Flask(__name__)

class LuckPillar(BaseModel):
    day: str = Field(..., description="Daily influence and its interpretation")
    day_influence: int = Field(..., description="Influence score for the day")

class TenGods(BaseModel):
    name: str = Field(..., description="The name of the Ten God")
    strength: str = Field(..., description="Strength level (e.g., Strong, Weak, Absent)")
    score: int = Field(..., description="Numerical influence score")
    description: str = Field(..., description="Detailed explanation of how this Ten God influences the person")

class Interaction(BaseModel):
    type: str = Field(..., description="Type of interaction (Clash, Combination, etc.)")
    description: str = Field(..., description="Detailed explanation of the interaction")
    emotional_keywords: List[str] = Field(..., description="Relevant emotional states")
    influence_score: int = Field(..., description="Influence score of the interaction")

class BaZiReport(BaseModel):
    day_master: str = Field(..., description="Day Master Element and its interpretation")
    current_luck_pillar: LuckPillar
    ten_gods: List[TenGods]
    notable_interactions: List[Interaction]
    personality: str = Field(..., description="Personality traits based on the BaZi chart")
    emotional_keywords: List[str] = Field(..., description="Overall emotional themes for the period")

def create_agents(readings, current_date):
    """Create and configure the agents with the provided readings and date"""
    
    # Agent 1: For creating a Bazi report
    report_expert = Agent(
        role="Bazi Report Expert",
        goal=f"Generate a report of today's fortune based on the Bazi chart content-{readings} for the date - {current_date}",
        backstory="""You are a skilled professional Bazi consultant, proficient in creating accurate and insightful reports. 
        Your task is to analyze user-provided Bazi chart readings and generate a personalized daily fortune report in a friendly, engaging, and easy-to-read format. 
        The report should be positive, entertaining, and practical, offering guidance on personality, career, relationships, health, wealth, and future outlook based on Bazi (四柱推命) principles. 
        Use clear language, relatable analogies, and actionable advice to ensure the insights are both meaningful and applicable to the user's daily life.""",
        allow_delegation=False,
        verbose=False,
        llm=llm
    )

    # Report Writing Task
    report_task = Task(
        description="""
        You will receive Bazi chart readings and generate a personalized daily fortune report in Japanese, ensuring it is very objective, clear and direct to the point and while maintaining accuracy and clarity. 
        Avoid uncommon, biased, or overly negative language, but remain honest—reporting both positive and challenging aspects without sugarcoating. 
        The report should be readily understandable, offering practical and encouraging guidance based on Bazi (四柱推命) principles. 
        Use a structured format with clear headings for each section, covering key aspects such as career, relationships, health, and wealth, while keeping the result reilable, factual and precise.
        The report provides an overview of the day's energies based on ten gods, notable interactions, and personality traits.  
        """,
        expected_output=""" 
            The final report should provide a personalized astrological reading for the current date based on the principles of Bazi from the given Bazi chart readings as input. 
            It identifies the individual's Day Master and provides a symbolic meaning for it, including potential strengths and weaknesses. 
            The report should display the current Luck Pillar (year, month, and day), giving symbolic meanings and influence scores for each. 
            It further details the "Ten Gods" for the day, indicating their strength and providing an interpretation of their combined influence. 
            Notable interactions, such as clashes and combinations between elements, are highlighted with their potential impact and associated emotions. 
            The report also offers insights into general life themes like career, relationships, health, and personal growth, providing summaries, detailed explanations, and influence scores for each. Finally, it outlines personality traits and overall emotional keywords.
            The final report should be strictly in japanese.
            It should follow the format and style similar to the example report below in japanese (The example report is delimited in 3 backticks). Example is for just taking reference of format.
            The whole report should be in japanese:

                ```{
            "day_master": "甲 (きのえ)",
            "current_luck_pillar": {
                "day": "乙巳 (きのとみ)",
                "day_influence": 7
            },
            "ten_gods": [
                {
                "name": "偏印 (へんいん)",
                "strength": "普通",
                "score": 5,
                "description": "今日はひらめき💡が冴える日！直感に従って動いてみると、新しい発見があるかも🤩 でも、ちょっと考えすぎちゃう傾向もあるから、深く考え込まずに、流れに身を任せてみるのもアリだよ😉"
                },
                {
                "name": "正財 (せいざい)",
                "strength": "やや強い",
                "score": 6,
                "description": "お金💰のこと、今日はしっかり管理できそう！無駄遣いせずに、計画的に使うのが吉🙆‍♀️ 将来のために貯金💰するのもいいね👍"
                },
                {
                "name": "食神 (しょくじん)",
                "strength": "弱い",
                "score": 3,
                "description": "今日はちょっと気分が乗らないかも…😔 でも、無理にテンション上げようとしないで、自然体でいるのが一番！美味しいもの🍴を食べたり、好きな音楽🎶を聴いたりして、リラックスする時間 を大切にね😌"
                },
                {
                "name": "傷官 (しょうかん)",
                "strength": "なし",
                "score": 0,
                "description": "今日はカッとなりやすい日🔥 怒りを感じても、グッとこらえて冷静に対応しよう！深呼吸🌬️して、心を落ち着かせることが大切だよ🧘‍♀️"
                },
                {
                "name": "比肩 (ひけん)",
                "strength": "やや強い",
                "score": 6,
                "description": "今日は自分のやりたいことを優先したくなる日🔥 周りの意見も大切だけど、自分の意思をしっかり持つことが重要！自信を持って行動すれば、きっとうまくいくよ💪"
                },
                {
                "name": "劫財 (ごうざい)",
                "strength": "なし",
                "score": 0,
                "description": "今日は周りの人とぶつかりやすい日💥 意見の衝突は避けられないかもだけど、感情的にならずに、冷静に話し合うことを心がけてね🤝"
                },
                {
                "name": "偏官 (へんかん)",
                "strength": "なし",
                "score": 0,
                "description": "今日はチャレンジ精神🔥が湧いてくる日！でも、無謀な行動は🙅‍♀️ 計画をしっかり立てて、慎重に行動することが成功の鍵🔑"
                },
                {
                "name": "正官 (せいかん)",
                "strength": "弱い",
                "score": 3,
                "description": "今日はルールを守って、真面目に行動するのが吉🙆‍♀️ でも、融通を利かせることも忘れずに😉"
                },
                {
                "name": "偏財 (へんざい)",
                "strength": "強い",
                "score": 8,
                "description": "今日はチャンス到来の予感✨ 積極的に行動して、幸運💰をつかもう！新しい出会い🤝や、 unexpectedな出来事が起こるかも…！？"
                },
                {
                "name": "正印 (せいいん)",
                "strength": "なし",
                "score": 0,
                "description": "今日は勉強📚に集中できる日！新しい知識を吸収して、スキルアップを目指そう🤓"
                }
            ],
            "notable_interactions": [
                {
                "type": "刑 (けい)",
                "description": "今日は何かとイライラしやすいかも…😠 周りの人に八つ当たりしないように気を付けてね⚠️ 気分転換に好きなことをして、リラックスしよう😌",
                "emotional_keywords": ["イライラ", "落ち着かない", "モヤモヤ"],
                "influence_score": 6
                },
                    {
                "type": "冲 (ちゅう)",
                "description": "今日は変化の兆しあり…！⚡️良い変化も悪い変化もあり得るから、落ち着いて対処しよう。焦らず、周りの意見にも耳を傾けてみて👂",
                "emotional_keywords": ["変化", "不安定", "落ち着かない"],
                "influence_score": 5
                }
            ],
            "personality": "あなたは、好奇心旺盛で行動力抜群のチャレンジャー！😆 新しいことにも物怖じせず、どんどん挑戦していく姿は周りの人をワクワクさせる✨ だけど、飽きっぽく、優柔不断なところもあるから 、そこだけ気を付けてね😉",
            "emotional_keywords": ["ワクワク", "好奇心", "ポジティブ"]
            }           
        ```""",
        output_pydantic=BaZiReport,
        agent=report_expert
    )

    # Agent 2: Content writer            
    content_generator = Agent(
        role="Today at a Glance writer",
        goal="Transform the summary into a engaging one-liner describing the day with witty language and Genz slangs while maintaining accuracy in Japanese",
        backstory=""" You're the Oscar Wilde of Bazi, the Shakespeare of Chinese astrology. 
        Your mission is to transform dry Bazi report into dazzling, Gen Z-approved one-liners in Japanese.
        You're a master wordsmith, a digital soothsayer, distilling ancient wisdom into bite-sized, meme-worthy pronouncements. 
        You work closely with the Bazi Report Expert and excel at maintaining the perfect balance between informative and entertaining writing, 
        ensuring the essence of the Bazi reading shines through while still bringing the LOLs.
        Think of yourself as the Bazi whisperer for the TikTok generation, delivering destiny with a side of sass.""",
        allow_delegation=False,
        verbose=False,
        llm=llm
    )

    # Content Writing Task
    content_task = Task(
        description="""Distill the given Bazi report into a single impactful sentence that is both informative and entertaining, tailored for Gen Z Japanese speakers using current internet slang.

            Guidelines:
            1.Understand the Essence
                -Identify the core message of the Bazi reading.
                -Highlight the most striking or defining aspect of the person's destiny.
                -The advice should be encouraging and uplifting, offering a fun and engaging perspective on their daily fortune.  
                -The advice should be honest without any sugar coating and negativity should also be reported whenever necessary.
            2.Make it Witty & Playful
                -Use humor, wordplay, or relatable scenarios to make it engaging.
                -Avoid generic advice—make it direct and snappy.
            3.Speak Gen Z
                -Incorporate recent (within the last two years) Japanese internet slang naturally.
                -Prioritize one-word slang over long phrases.
                -Ensure the slang is currently trending (avoid outdated words like まじ卍).
                -Ensure it sounds like a casual message from a Gen Z friend.
                -Avoid using slang awkwardly—if standard Japanese fits better, use it.
            4.Be Accurate Yet Entertaining
                -While being witty, ensure the sentence stays true to the Bazi summary.
                -No abstract metaphors—keep it clear and easy to understand.
            5.Keep it Concise
                -The output must be one sentence only—no punctuation.
                -It should be intelligent, creative, and thought-provoking.
                -The sentence should not exceed 10 words in japanese text format.
            6.Maintain a Respectful Tone
                -Avoid offensive, inappropriate, or biased language.
                -Keep it constructive, neutral, and positive.
            7.The result should be not specific to any gender""",
        expected_output="""The output should be a single sentence, written in Japanese without any translation, that fulfills the following criteria:
        -Witty and Humorous: It should make the reader chuckle or at least crack a smile.
        -Gen Z Slang: It should incorporate current internet slang and trends (used tastefully).
        -Informative: It should convey some key insights from the Bazi chart, like personality traits or life path.
        -Concise: A single, impactful sentence.
        -Examples (based on a hypothetical Bazi summary, note these are still in English to illustrate the tone desired, your output must be in Japanese): 
             1."Don't hold yourself back"
             2."Explore your social side"
             3."Do not overthink", 
             4."Listen to something other than sad songs today"
             5."Clinging isn't the best answer" 
             6."Let yourself be a bird today" 
             7."Spend time getting to know your other half"
        -Avoid symbols which are not used in casual Japanese by Genz and Milennialls like "卍".
        -End the sentence with the required punctuation mark based on "Today at a Glance."
        - Use the examples as inspiration for tone and style, but don't just rehash them. Get creative, invent new phrases, and tailor the slang to the specific chart's characteristics (e.g., hardworking, social, creative). 
          Balance humor with clarity—the meaning should still be accessible. Think ironic, self-aware, and a touch absurd. Surprise me!""",
        agent=content_generator
    )

    # Create Crew
    crew = Crew(
        agents=[report_expert, content_generator],
        tasks=[report_task, content_task],
        verbose=True
    )
    
    return crew

@app.route('/api/bazi-report', methods=['POST'])
def generate_bazi_report():
    """API endpoint to generate BaZi report"""
    
    # Extract readings from raw text input
    readings = request.data.decode('utf-8')  # Read plain text input
    
    # If readings are empty, return an error
    if not readings.strip():
        return jsonify({"error": "No readings provided"}), 400
    
    # Get current date
    today_date = date.today()
    current_date = today_date.strftime("%Y-%m-%d")
    
    # Create agents and run the tasks
    crew = create_agents(readings, current_date)
    result = crew.kickoff(inputs={"readings": readings, "current_date": current_date})
    
    # Return the generated report
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)