from get_policy import policy_agent
myagent=policy_agent("/home/xu/code/CS_RL_xu/model/Qwen3-0.6B")
que=input("输入当前状态")
choice="A.向前走,B.向后转,C.向右转,D.向左走,E.开火"
ans=myagent.getans(que,choice)
#print(tans)
print(ans)
