from abc import ABC, abstractmethod

class Coffee(ABC):
    """
    Interface for creating various coffee types
    """
    @abstractmethod
    def make_coffee(self):
        pass


# Implements concrete coffee classes
class Espresso(Coffee):
    """
    Implements Espresso coffee
    """

    def make_coffee(self):
        return f"Making espresso for {self}"
    
    
class Cappucino(Coffee):
    """
    Implements Cappucino 
    """

    def make_coffee(self):
        return f"Making Cappucino for {self}"
    

class Latte(Coffee):
    """
    Implements Latte 
    """

    def make_coffee(self):
        return f"Making latte for {self}"
    

class CoffeeMachine:
    """
    hjjkdjkmkdmc
    """
    def get_coffee(self, coffee_type: str):
        if coffee_type == "Espresso":
            return Espresso().make_coffee()

        elif coffee_type == "Cappucino":
            return Cappucino().make_coffee()
        
        elif coffee_type == "Latte":
            return Latte().make_coffee()

        else:
            raise ValueError("Invalid coffee type")


if __name__=="__main__":
    machine = CoffeeMachine()
    print(machine.get_coffee("Latte"))