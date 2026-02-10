
class SimpleAgent:
    """
    A simple demonstration of an AI agent's core loop
    """
    def __init__(self, name="Agent"):
        self.name = name
        self.state = {"observations": [], "actions": [], "goals": []}
        self.tools = {
            "search": self._search_tool,
            "calculate": self._calculate_tool,
            "store": self._store_tool
        }
    
    def _search_tool(self, query):
        """Simulated search tool"""
        return f"Search results for: {query}"
    
    def _calculate_tool(self, expression):
        """Simulated calculator tool"""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error: Invalid expression"
    
    def _store_tool(self, key, value):
        """Simulated storage tool"""
        self.state[key] = value
        return f"Stored {key} = {value}"
    
    def observe(self, observation):
        """Agent observes the environment"""
        self.state["observations"].append(observation)
        print(f"[{self.name}] Observing: {observation}")
        return observation
    
    def think(self, goal):
        """Agent reasons about what action to take"""
        print(f"[{self.name}] Thinking about goal: {goal}")
        
        # Simple reasoning logic (in real agents, this would use an LLM)
        if "search" in goal.lower():
            return "search", goal.replace("search", "").strip()
        elif "calculate" in goal.lower() or any(op in goal for op in ["+", "-", "*", "/"]):
            return "calculate", goal
        elif "store" in goal.lower():
            parts = goal.split()
            if len(parts) >= 3:
                return "store", (parts[1], parts[2])
        return None, None
    
    def act(self, tool_name, tool_input):
        """Agent executes an action"""
        if tool_name in self.tools:
            if isinstance(tool_input, tuple):
                result = self.tools[tool_name](*tool_input)
            else:
                result = self.tools[tool_name](tool_input)
            self.state["actions"].append((tool_name, tool_input, result))
            print(f"[{self.name}] Acting: {tool_name}({tool_input}) -> {result}")
            return result
        else:
            print(f"[{self.name}] Unknown tool: {tool_name}")
            return None
    
    def reflect(self, action_result):
        """Agent reflects on the outcome"""
        print(f"[{self.name}] Reflecting: Action completed, result: {action_result}")
        return action_result
    
    def run_loop(self, goal, observation=None):
        """Execute the complete agent loop"""
        print(f"\n{'='*50}")
        print(f"Agent Loop Execution: {goal}")
        print(f"{'='*50}\n")
        
        # 1. Observe
        if observation:
            self.observe(observation)
        
        # 2. Think
        tool_name, tool_input = self.think(goal)
        
        if tool_name:
            # 3. Act
            result = self.act(tool_name, tool_input)
            
            # 4. Reflect
            self.reflect(result)
            return result
        else:
            print(f"[{self.name}] Could not determine action for goal: {goal}")
            return None



# Example: Multi-Agent System



class MultiAgentSystem:
    def __init__(self):
        self.agents = {
            "researcher": SimpleAgent("Researcher"),
            "analyst": SimpleAgent("Analyst"),
            "writer": SimpleAgent("Writer")
        }
    
    def collaborate(self, task):
        print(f"\n{'='*60}")
        print(f"Multi-Agent Collaboration: {task}")
        print(f"{'='*60}\n")
        
        # Researcher gathers information
        research_result = self.agents["researcher"].run_loop(
            f"search for {task}", 
            observation=f"Task assigned: {task}"
        )
        
        # Analyst processes the information
        analysis_result = self.agents["analyst"].run_loop(
            f"calculate analysis of {task}",
            observation=f"Research completed: {research_result}"
        )
        
        # Writer creates output
        writer_result = self.agents["writer"].run_loop(
            f"store report {task}",
            observation=f"Analysis completed: {analysis_result}"
        )
        
        return {
            "research": research_result,
            "analysis": analysis_result,
            "report": writer_result
        }

# Example
multi_system = MultiAgentSystem()
result = multi_system.collaborate("climate change impacts")
class SimpleAgent:
    """
    A simple demonstration of an AI agent's core loop
    """
    def __init__(self, name="Agent"):
        self.name = name
        self.state = {"observations": [], "actions": [], "goals": []}
        self.tools = {
            "search": self._search_tool,
            "calculate": self._calculate_tool,
            "store": self._store_tool
        }
    
    def _search_tool(self, query):
        """Simulated search tool"""
        return f"Search results for: {query}"
    
    def _calculate_tool(self, expression):
        """Simulated calculator tool"""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error: Invalid expression"
    
    def _store_tool(self, key, value):
        """Simulated storage tool"""
        self.state[key] = value
        return f"Stored {key} = {value}"
    
    def observe(self, observation):
        """Agent observes the environment"""
        self.state["observations"].append(observation)
        print(f"[{self.name}] Observing: {observation}")
        return observation
    
    def think(self, goal):
        """Agent reasons about what action to take"""
        print(f"[{self.name}] Thinking about goal: {goal}")
        
        # Simple reasoning logic (in real agents, this would use an LLM)
        if "search" in goal.lower():
            return "search", goal.replace("search", "").strip()
        elif "calculate" in goal.lower() or any(op in goal for op in ["+", "-", "*", "/"]):
            return "calculate", goal
        elif "store" in goal.lower():
            parts = goal.split()
            if len(parts) >= 3:
                return "store", (parts[1], parts[2])
        return None, None
    
    def act(self, tool_name, tool_input):
        """Agent executes an action"""
        if tool_name in self.tools:
            if isinstance(tool_input, tuple):
                result = self.tools[tool_name](*tool_input)
            else:
                result = self.tools[tool_name](tool_input)
            self.state["actions"].append((tool_name, tool_input, result))
            print(f"[{self.name}] Acting: {tool_name}({tool_input}) -> {result}")
            return result
        else:
            print(f"[{self.name}] Unknown tool: {tool_name}")
            return None
    
    def reflect(self, action_result):
        """Agent reflects on the outcome"""
        print(f"[{self.name}] Reflecting: Action completed, result: {action_result}")
        return action_result
    
    def run_loop(self, goal, observation=None):
        """Execute the complete agent loop"""
        print(f"\n{'='*50}")
        print(f"Agent Loop Execution: {goal}")
        print(f"{'='*50}\n")
        
        # 1. Observe
        if observation:
            self.observe(observation)
        
        # 2. Think
        tool_name, tool_input = self.think(goal)
        
        if tool_name:
            # 3. Act
            result = self.act(tool_name, tool_input)
            
            # 4. Reflect
            self.reflect(result)
            return result
        else:
            print(f"[{self.name}] Could not determine action for goal: {goal}")
            return None

# Example usage
#agent = SimpleAgent("DemoAgent")

# Example 1: Search
#agent.run_loop("search for Python tutorials", observation="User wants to learn Python")

#print("\n")

# Example 2: Calculate
#agent.run_loop("calculate 15 * 23 + 7", observation="User needs a calculation")

#print("\n")

# Example 3: Store
#agent.run_loop("store name Alice", observation="User wants to save information")

# print("\n")
# print("Final Agent State:")
# print(agent.state)
