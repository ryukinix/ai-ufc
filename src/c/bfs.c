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
#include "graph.h"
#define N 1000

static int path[N];

void init(int v[], int n) {
    for (int i = 0; i < n; i++) {
        v[i] = Z;
    }
}

void print_vec(int v[], int n) {
    for (int i = 0; i < n && v[i] != Z; i++) {
        printf("%c ", vertex_label(v[i]));
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
    printf("=> SEARCH: (%c, %c)\n", vertex_label(s), vertex_label(g));
    // variables
    int lookup[V_MAX]; // fila
    int lookup_size = 0;
    int memory[V][V]; // backtracking
    int local_path[V];


    // initialziation
    init(local_path, V);
    init(lookup, V_MAX);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            memory[i][j] = i == j? 0 : Z;
        }
    }
    // breadth first search of (s, g) on matrix graph
    Vertex current_node =  s;
    Vertex last_node = s;
    //printf("NODES: %d ", s);
    puts(":: START SEARCH");
    do {
        for(int i = 0; i < V; i++) {
            int neighbor = graph[i][current_node];
            if(neighbor > 0 && memory[i][current_node] == Z) {
                printf("%c->%c\n", vertex_label(current_node), vertex_label(i));
                queue_push(lookup, V_MAX, i);
                lookup_size++;
                memory[i][current_node] = last_node+1;
                memory[current_node][i] = last_node+1;
            }
        }
        last_node = current_node;
        current_node = queue_pop(lookup, V_MAX);
        lookup_size--;
    } while (current_node != g);
    puts(":: END SEARCH");

    puts("=> MEMORY");


    printf("   ");
    for(int i = 0; i < V; i++) {
        printf("%c  ", vertex_label(i));
    }
    puts("");


    for (int i = 0; i < V; i++) {
        printf("%c ", vertex_label(i));
        for (int j = 0; j < V; j++) {
            int v = memory[i][j];
            if (v == 0) {
                printf("%2d ", v);
            } else {
                printf("%2c ", vertex_label(v-1));
            }
        }
        puts("");
    }

    // backtracking
    puts(":: START BACKTRACKING");
    local_path[0] = g;
    int k = 1;
    while (current_node != s) {
        for (int i = 0; i < V; i++) {
            int cost = memory[current_node][i];
            if (cost > 0 && !vec_in(local_path, V, i)) {
                local_path[k] = i;
                current_node = i;
                break;
            }
        }
        printf("=> %c->%c\n", vertex_label(local_path[k-1]), vertex_label(local_path[k]));
        k++;
    }
    puts(":: END BACKTRACKING");

    puts(":: START PATH REVERSE");
    for(int j = 0; j < k; j++) {
        path[j] = local_path[k - j - 1];
    }
    puts(":: END PATH REVERSE");
}

int main(void) {
    Vertex s = H;
    Vertex g = C;
    init(path, N);
    bfs(s, g);
    printf("PATH: ");
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
