romenia = {
    "bucareste": [
        ["giurgiu", 90],
        ["urziceni", 85],
        ["fagaras", 211],
        ["pitesti", 101],
    ],
    "urziceni": [["bucareste", 85], ["hirsova", 98], ["vaslui", 142]],
    "hirsova": [["urziceni", 98], ["eforie", 86]],
    "eforie": [["hirsova", 86]],
    "vaslui": [["urziceni", 142], ["lasi", 92]],
    "lasi": [["vaslui", 92], ["neamt", 87]],
    "neamt": [["lasi", 87]],
    "giurgiu": [["bucareste", 90]],
    "pitesti": [["craiova", 138], ["rimnieu vilcea", 97], ["bucareste", 101]],
    "fagaras": [["sibiu", 99], ["bucareste", 211]],
    "craiova": [["pitesti", 138], ["rimnieu vilcea", 146], ["dobreta", 120]],
    "rimnieu vilcea": [["pitesti", 97], ["craiova", 146], ["sibiu", 80]],
    "sibiu": [["fagaras", 99],
              ["rimnieu vilcea", 80],
              ["oradea", 151],
              ["arad", 140]],
    "dobreta": [["craiova", 120], ["mehadia", 75]],
    "mehadia": [["dobreta", 75], ["lugoj", 70]],
    "lugoj": [["mehadia", 70], ["timisoara", 111]],
    "timisoara": [["lugoj", 111], ["arad", 118]],
    "arad": [["timisoara", 118], ["sibiu", 140], ["zerind", 75]],
    "zerind": [["arad", 75], ["oradea", 71]],
    "oradea": [["zerind", 71], ["sibiu", 151]],
}
