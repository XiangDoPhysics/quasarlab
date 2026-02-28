from main import run_agent

if __name__ == "__main__":
    query = input("QUASARlab > ")
    response = run_agent(query)
    print(response)