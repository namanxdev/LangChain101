import os
import langchain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain101!")
    information = """
    Mukesh Ambani set up Reliance Infocomm Limited (later Reliance Communications Limited), which was focused on information and communications technology initiatives.[28] At the age of 24, Ambani was given charge of the construction of Patalganga petrochemical plant when the company was heavily investing in oil refinery and petrochemicals.[29]

Ambani directed and led the creation of the world's largest grassroots petroleum refinery in Jamnagar, India, which could produce 660,000 barrels per day (33 million tonnes per year) in 2010, integrated with petrochemicals, power generation, port, and related infrastructure.[30] In December 2013, Ambani announced the possibility of a "collaborative venture" with Bharti Airtel in setting up digital infrastructure for the 4G network in India.[31] On 18 June 2014, Ambani said he will invest Rs 1.8 trillion (short scale) across businesses in the next three years and launch 4G broadband services in 2015.[32]


United States Secretary of State Hillary Clinton with Ratan Tata of the Tata Group (left) and Ambani (right) in July 2009.
In February 2014, a First Information Report (FIR) alleging criminal offences was filed against Ambani for alleged irregularities in the pricing of natural gas from the KG basin.[33]

Ambani was elected as a member of the National Academy of Engineering in 2016 for engineering and business leadership in oil refineries, petrochemical products, and related industries.[34]

As of 2015, Ambani ranked fifth among India's philanthropists, according to China's Hurun Research Institute.[35] He was appointed as a Director of Bank of America and became the first non-American to be on its board.[36] As of 2016, Ambani was ranked as the 36th richest person in the world and has consistently held the title of India's richest person on Forbes magazine's list for the past ten years.[37] He is the only Indian businessman on Forbes' list of the world's most powerful people.[38] He surpassed Jack Ma, executive chairman of Alibaba Group,[39] to become Asia's richest person with a net worth of $44.3 billion in July 2018.[40] He is also the wealthiest person in the world outside North America and Europe.[41]

As of February 2018, Bloomberg's "Robin Hood Index" estimated that Ambani's personal wealth was enough to fund the operations of the Indian federal government for 20 days.[42]

Through Reliance, Ambani also owns the Indian Premier League franchise Mumbai Indians and is the founder of the Indian Super League, a football league in India.[43]

Reliance Industries has faced criticism for maintaining business relations with Russia despite international sanctions imposed following Russia’s invasion of Ukraine in 2022.[44][45] India imported 42% of its oil from Russia in 2025, up from 3% in 2021, with Reliance being the largest importer.[46
    """

    Summary_template = """
You are a helpful assistant that summarizes the following information:
given the information {information} of a person i want you to create:
1. A short summary of the person's life
2. 2 interesting facts about the person


"""

    Summary_prompt = PromptTemplate(input_variables=["information"], template=Summary_template)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # llm = ChatOllama(model="deepseek-r1:1.5b", temperature=5)
    chain = Summary_prompt | llm
    response = chain.invoke({"information": information})
    print(response.content)
if __name__ == "__main__":
    main()
