from get_policy import policy_agent
myagent=policy_agent("/home/xu/code/CS_RL_xu/model/Qwen3-0.6B")
que=input("输入当前状态")
ans=myagent.getans(que)
#print(tans)
print(ans)
