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

#include "ds-ufc/ds-ufc.h"

int main(void) {
    Stack *s = stack_create();
    stack_push(s, 10);
    stack_push(s, 20);
    stack_push(s, 30);
    stack_println(s);

    return 0;
}
