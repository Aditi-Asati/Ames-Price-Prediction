from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    """
    Interface for making a payment
    """
    @abstractmethod
    def pay(self, amount: float):
        pass

class BitcoinPayment(PaymentStrategy):
    """
    Implements bitcoin payment 
    """

    def pay(self, amount: float):
        return f"Making payment of {amount} Rs via bitcoin"
    

class CreditCardPayment(PaymentStrategy):
    """
    implements credit card payment
    """

    def pay(self, amount: float):
        return f"making payment of {amount} Rs via credit card"
    
class UPIPayment(PaymentStrategy):
    """
    implements UPI payment
    """

    def pay(self, amount: float):
        return f"making payment of {amount} Rs via UPI"
    
class Transaction:
    """
    Make a transaction using some payment
    """

    def __init__(self, payment_strategy: PaymentStrategy) -> None:
        self.strategy = payment_strategy

    def checkout(self, amount: float):
        return self.strategy.pay(amount)
    

        
if __name__=="__main__":
    transaction = Transaction(UPIPayment())
    print(transaction.checkout(300))


