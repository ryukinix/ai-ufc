#define V 9
#define V_MAX (1 << 9)
#define Z -1 // not exists


typedef enum vertex {
    A, B, C, D, E, F, G, H, I
} Vertex;


int graph[V][V] = {
    {0, 1, Z, Z, Z, 1, Z, Z, Z}, // A
    {1, 0, 1, 1, Z, Z, Z, Z, Z}, // B
    {Z, 1, 0, 1, Z, Z, 2, Z, Z}, // C
    {Z, 1, 1, 0, 1, Z, Z, Z, Z}, // D
    {Z, Z, Z, 1, 0, Z, Z, 1, Z}, // E
    {1, Z, Z, Z, Z, 0, Z, Z, 1}, // F
    {Z, Z, 2, Z, 1, Z, 0, Z, Z}, // G
    {Z, Z, Z, Z, 1, Z, Z, 0, 1}, // H
    {Z, Z, Z, Z, Z, 1, Z, 1, 0}, // I
    /*A B  C  D  E  F  G  H  I*/
};

static inline char vertex_label(int k) {
    switch (k) {
    case 0: return 'A';
    case 1: return 'B';
    case 2: return 'C';
    case 3: return 'D';
    case 4: return 'E';
    case 5: return 'F';
    case 6: return 'G';
    case 7: return 'H';
    case 8: return 'I';
    default: return 'X';
    }
}
