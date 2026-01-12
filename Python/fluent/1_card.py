import collections
from random import choice

Card = collections.namedtuple("Card", ["rank", "suit"])


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = "spades diamonds clubs hearts".split()

    suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

    @staticmethod
    def spades_high(card):
        rank_value = FrenchDeck.ranks.index(card.rank)
        return (
            rank_value * len(FrenchDeck.suit_values) + FrenchDeck.suit_values[card.suit]
        )


if __name__ == "__main__":
    print("=" * 20, "beer_card : ", "=" * 20)
    beer_card = Card("7", "diamonds")
    print(beer_card)
    print("=" * 40)

    print("=" * 20, "FrenchDeck 생성", "=" * 20)
    deck = FrenchDeck()
    print("한 세트 내 카드 갯수 : ", len(deck))

    print("첫 카드 = ", deck[0])
    print("마지막 카드 = ", deck[-1])
    # for i in range(52):
    #     print(f"{i} 번째 카드 , {deck[i]}")
    # for card in deck:
    #     print(card)
    # for card in reversed(deck):
    #     print(card)

    print("=" * 20, "random pick !", "=" * 20)
    print(choice(deck))
    print(choice(deck))
    print(choice(deck))

    print("=" * 20, "ACE 뽑기 !!", "=" * 20)
    print(deck[12::13])

    print("=" * 20, "오름차순 정렬", "=" * 20)
    for card in sorted(deck, key=FrenchDeck.spades_high):
        print(card)

    print("=" * 40)
