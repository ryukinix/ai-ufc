/**
 * ================================================
 *
 *         Copyright 2019 Manoel Vilela
 *
 *         Author: Manoel Vilela
 *        Contact: manoel_vilela@engineer.com
 *   Organization: UFC
 *
 * ===============================================
 */

// #include "ds-ufc/ds-ufc.h"

#include <stdio.h>
#include <stdlib.h>
#define N 1000
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

static int path[N];

char vertex_label(int k) {
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


void init(int v[], int n) {
    for (int i = 0; i < n; i++) {
        v[i] = Z;
    }
}

void print_vec(int v[], int n) {
    for (int i = 0; i < n && v[i] != Z; i++) {
        printf("\n%d -> %c", i, vertex_label(v[i]));
    }
    puts("");
}

int vec_in(int v[], int n, int k) {
    for (int i = 0; i < n && v[i] != Z; i++) {
        if (v[i] == k) {
            return 1;
        }
    }
    return 0;
}

void queue_push(int v[], int n, int value) {
    int i;
    if (v[0] == Z) {
        v[0] = value;
    } else {
        for(i = 0; i < n && v[i] != Z; i++);
        v[i] = value;
    }

}

int queue_pop(int v[], int n) {
    int i;
    int value = v[0];

    for(i = 1; i < n - 1 && v[i] != -1; i++) {
        v[i - 1] = v[i];
    }

    v[i] = Z;
    return value;
}

int heap_parent(int n) {
    return n/2;
}


int heap_left_child(int n) {
    return 2 * n;
}


int heap_right_child(int n) {
    return 2 * n + 1;
}


void bfs(Vertex s, Vertex g) {

    // variables
    int lookup[V_MAX]; // fila
    int memory[V_MAX]; // fila
    int lookup_size = 0;
    int memory_size = 0;
    int local_path[V];

    // initialziation
    init(local_path, V);
    init(lookup, V_MAX);
    init(memory, V_MAX);
    // breadth first search of (s, g) on matrix graph
    Vertex current_node =  s;
    memory[0] = s;
    //printf("NODES: %d ", s);
    do {
        for(int i = 0; i < V; i++) {
            int neighbor = graph[i][current_node];
            if(neighbor > 0 && !vec_in(memory, V_MAX, i)) {
                //printf("(%d->%d)\n", current_node, i);
                queue_push(lookup, V_MAX, i);
                lookup_size++;
            }
        }
        current_node = queue_pop(lookup, V_MAX);
        lookup_size--;
        if (!vec_in(memory, V_MAX, (int) current_node)) {
            queue_push(memory, V_MAX, current_node);
            memory_size++;
        }
        //printf("%d ", current_node);
    } while (current_node != g);
    puts("");

    int heapp = memory_size;
    int i;
    local_path[0] = memory[heapp];
    for(i = 1; heapp > 0; i++) {
        heapp /= 2;
        local_path[i] = memory[heapp];
    }

    for(int j = 0; j <= i; j++) {
        path[j] = local_path[i - j - 1];
    }

    printf("MEMORY SIZE: %d\n", memory_size );
    printf("MEMORY: ");
    print_vec(memory, V_MAX);

    /* printf("LOOKUP SIZE: %d\n", lookup_size ); */
    /* printf("LOOKUP: "); */
    /* print_vec(lookup, V_MAX); */
}

int main(void) {
    Vertex s = H;
    Vertex g = C;
    init(path, N);
    bfs(s, g);
    printf("=> PATH: (%c, %c)\n", vertex_label(s), vertex_label(g));
    for(int i = 0; path[i] != Z; i++) {
        printf("%c ", vertex_label(path[i]));
    }
    puts("");
    // EXPECTED:
    //  VERTEX: 7
    //  VERTEX: 4
    //  VERTEX: 3
    //  VERTEX: 2 | 6
    return 0;
}
