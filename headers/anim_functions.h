#ifndef __ANIM_FUNCTIONS_H__
#define __ANIM_FUNCTIONS_H__

void update_params(CA *ca, bool *keys)
{
    unsigned i, carcinIdx, deactivated;
    bool activate = false;
    char answer[11] = { '\0' };

    if (keys[0])
        ca->save ? ca->save = false : ca->save = true;
    else if (keys[1])
        user_input("Enter how many time steps TC are alive: ", "%d", &ca->maxTTCAlive, -1, ca->maxT);
    else if (keys[2])
        ca->perfectExcision ? ca->perfectExcision = false
                            : ca->perfectExcision = true;
    else if (keys[3]) {
        user_input("Do you want to activate or deactivate? ", "%s", answer, 10, "deactivate,activate");
        if (strcmp("activate", answer) == 0) activate = true;
        user_input("Enter how many carcinogens (0-%d): ", "%d", ca->maxNCarcin,
                   &ca->nCarcin, 0, ca->maxNCarcin);
        if (ca->nCarcin == 0) {
            do {
                user_input("Enter a carcinogen index (0-%d): ", "%d", ca->maxNCarcin-1,
                           &carcinIdx, 0, ca->maxNCarcin-1);
                if (carcinIdx < ca->maxNCarcin) {
                    ca->activeCarcin[carcinIdx] = activate;
                    if (activate) ca->nCarcin++;
                    else {
                        ca->carcins[carcinIdx].t = 0;
                        ca->carcins[carcinIdx].nCycles = 0;
                    }
                }
                user_input("Do you want to enter another carcinogen index? (yes/no) ", "%s",
                           answer, 3, "no,yes");
            } while (strcmp("no", answer) != 0);
        } else if (ca->nCarcin > 0) {
            deactivated = 0;
            for (i = 0; i < ca->nCarcin; i++)
                if (i < ca->maxNCarcin) {
                    ca->activeCarcin[i] = activate;
                    if (!activate) {
                        deactivated++;
                        ca->carcins[i].t = 0;
                        ca->carcins[i].nCycles = 0;
                    }
                } else ca->nCarcin = ca->maxNCarcin;
            if (!activate) ca->nCarcin -= deactivated;
        }
    } else if (keys[4]) {
        do {
            carcinIdx = carcin_param_change(ca->maxNCarcin, "Enter the number of time steps influx occurs: ",
                                            "%d", &ca->carcins[carcinIdx].maxTInflux, -1, INT_MAX);
            if (carcinIdx < ca->maxNCarcin && ca->carcins[carcinIdx].maxTInflux != -1)
                ca->carcins[carcinIdx].maxTInflux *= ca->carcins[carcinIdx].exposureTime;
            user_input("Do you want to enter another carcinogen index? (yes/no) ", "%s", answer, 3, "no,yes");
        } while (strcmp("no", answer) != 0);
    } else if (keys[5]) {
        do {
            carcinIdx = carcin_param_change(ca->maxNCarcin, "Enter the number of time steps no influx occurs: ",
                                            "%d", &ca->carcins[carcinIdx].maxTNoInflux, -1, INT_MAX);
            if (carcinIdx < ca->maxNCarcin && ca->carcins[carcinIdx].maxTNoInflux != -1)
                ca->carcins[carcinIdx].maxTNoInflux *= ca->carcins[carcinIdx].exposureTime;
            user_input("Do you want to enter another carcinogen index? (yes/no) ", "%s", answer, 3, "no,yes");
        } while (strcmp("no", answer) != 0);
    } else if (keys[6]) {
        do {
            carcin_param_change(ca->maxNCarcin, "Enter the exposure time period (hours): ", "%lf",
                                &ca->carcins[carcinIdx].exposureTime, 0.000277778, DBL_MAX);
            user_input("Do you want to enter another carcinogen index? (yes/no) ", "%s", answer, 3, "no,yes");
        } while (strcmp("no", answer) != 0);
    } else if (keys[7]) {
        do {
            carcin_param_change(ca->maxNCarcin, "Enter the sensitivity function index: ", "%u",
                                &ca->carcins[carcinIdx].funcIdx, 0, ca->carcins[carcinIdx].nFunc - 1);
            user_input("Do you want to enter another carcinogen index? (yes/no) ", "%s", answer, 3, "no,yes");
        } while (strcmp("no", answer) != 0);
    } else if (keys[8]) {
        do {
            carcin_param_change(ca->maxNCarcin, "Enter the carcinogen function type (0-2): ", "%u",
                                &ca->carcins[carcinIdx].type, 0, 2);
            user_input("Do you want to enter another carcinogen index? (yes/no) ", "%s", answer, 3, "no,yes");
        } while (strcmp("no", answer) != 0);
    }
}

unsigned* update_graph_data(CA *ca, unsigned ticks, unsigned updateType,
                      unsigned *countData=NULL, unsigned *countKills=NULL,
                      unsigned *countTAC=NULL, cudaStream_t *stream=NULL)
{
    size_t unsgn = sizeof(unsigned), dbl = sizeof(double);
    unsigned i, j, step, count, *totals, *countDataTemp;
    char **totalFname, fname[100] = { '\0' };
    dim3 colour, blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                        NBLOCKS(ca->gridSize, ca->blockSize)),
         threads(ca->blockSize, ca->blockSize);
    double *phenoSum, *geneExprSum, *nPosMutSum, *fitnessSum,
           numerator1, numerator2, denom, denom1, denom2, total;

    // Update files related to dedifferentiation and cells that changed state
    if (updateType == 0) {
        totals = (unsigned*)malloc(2*unsgn);

        save_count_data("numChangeState.data", ca->headerCount,
                        ticks, countData[EMPTY], 0, 0, 0);
        save_count_data("numChangeState_toMutated.data", ca->headerCount,
                        ticks, countData[NC] + countData[SC] + countData[TC],
                        0, 0, 0);
        save_count_data("numChangeState_fromMutated.data", ca->headerCount,
                        ticks, countData[MNC] + countData[MSC], 0, 0, 0);
        save_count_data("numDeDiff_SC.data", ca->headerCount,
                        ticks, countData[31], 0, 0, 0);
        save_count_data("numDeDiff_Empty.data", ca->headerCount,
                        ticks, countData[32], 0, 0, 0);
        save_count_data("numDeDiff_Both.data", ca->headerCount,
                        ticks, countData[33], 0, 0, 0);
        save_count_data("numDeDiff_Rand.data", ca->headerCount,
                        ticks, countData[34], 0, 0, 0);
        save_count_data("numDeDiff.data", ca->headerCount, ticks,
                        countData[31] + countData[32] +
                        countData[33] + countData[34], 0, 0, 0);
        for (i = 0; i < ca->nStates-1; i++) {
            sprintf(fname, "numChangeState_State%d.data", i);
            colour.x = ca->stateColors[i].x;
            colour.y = ca->stateColors[i].y;
            colour.z = ca->stateColors[i].z;
            if (i == CSC) {
                colour.x = ca->stateColors[MSC].x;
                colour.y = ca->stateColors[MSC].y;
                colour.z = ca->stateColors[MSC].z;
            } else if (i == TC) {
                colour.x = ca->stateColors[SC].x;
                colour.y = ca->stateColors[SC].y;
                colour.z = ca->stateColors[SC].z;
            }
            save_count_data(fname, ca->headerCount,
                            ticks, countData[i], colour.x,
                            colour.y, colour.z);

            if (i == SC || i == MSC || i == CSC) continue;
            step = 0;
            if (i == MNC) step = 8;
            else if (i == TC) step = 16;
            memset(totals, 0, 2*unsgn);
            for (j = 0; j < 15; j++) {
                sprintf(fname, "numDeDiff_State%d", i);
                if (j < 4 || j == 12) strcat(fname, "_TAC");
                else if (j < 8 || j == 13) strcat(fname, "_Non-TAC");
                if (j == 0 || j == 4 || j == 8) strcat(fname, "_SC.data");
                else if (j == 1 || j == 5 || j == 9) strcat(fname, "_Empty.data");
                else if (j == 2 || j == 6 || j == 10) strcat(fname, "_Both.data");
                else if (j == 3 || j == 7 || j == 11) strcat(fname, "_Rand.data");
                else strcat(fname, ".data");
                if (j < 8) count = countData[step+ca->nStates+j];
                if (j > 7 && j < 12) count = countData[step+ca->nStates+(j-8)]
                                           + countData[step+ca->nStates+(j-8)+4];
                if (j == 12) count = totals[0];
                if (j == 13) count = totals[1];
                if (j == 14) count = totals[0] + totals[1];
                save_count_data(fname, ca->headerCount,
                                ticks, count, ca->stateColors[i].x,
                                ca->stateColors[i].y, ca->stateColors[i].z);
                if (j < 4) totals[0] += countData[step+ca->nStates+j];
                else if (j < 8) totals[1] += countData[step+ca->nStates+j];
            }
        }

        free(totals);
    // TAC and number of cells killing other cells in various situations
    } else if (updateType == 1) {
        totals = (unsigned*)malloc(12*unsgn);
        memset(totals, 0, 12*unsgn);
        totalFname = (char**)malloc(9*sizeof(char*));
        for (i = 0; i < 9; i++) totalFname[i] = (char*)calloc(50, 1);

        for (i = 0; i < 4; i++) {
            save_count_data(ca->countFiles[i+ca->nStates+4], ca->headerCount, ticks,
                            countData[i], 0, 0, 0);
            for (j = 0; j < ca->nStates; j++) {
                if (j == EMPTY || (i == DIFF && (j == NC || j == MNC || j == TC)))
                    continue;
                if (i != DIFF) {
                    save_count_data(ca->countFiles[j*3+(ca->nGenes+15)+i],
                                    ca->headerCount, ticks,
                                    countData[(j*3+4)+i], ca->stateColors[j].x,
                                    ca->stateColors[j].y, ca->stateColors[j].z);
                    if (j == NC || j == MNC || j == TC) {
                        step = 0;
                        if (j == TC) step = 3;
                        sprintf(fname, "numTAC_Pheno%d_State%d.data", i, j);
                        save_count_data(fname, ca->headerCount, ticks,
                                        countTAC[(j-step)*3+i], ca->stateColors[j].x,
                                        ca->stateColors[j].y, ca->stateColors[j].z);
                        totals[9+i] += countTAC[(j-step)*3+i];
                    }
                } else
                    save_count_data(ca->countFiles[j+(3*ca->nStates+ca->nGenes+10)],
                                    ca->headerCount, ticks,
                                    countData[(3*ca->nStates-1)+j], ca->stateColors[j].x,
                                    ca->stateColors[j].y, ca->stateColors[j].z);
            }
            if (i == DIFF) continue;
            sprintf(fname, "numTAC_Pheno%d.data", i);
            save_count_data(fname, ca->headerCount, ticks,
                            totals[9+i], 0, 0, 0);
        }
        for (i = 0; i < ca->nStates-1; i++) {
            for (j = 0; j < ca->nStates; j++) {
                // Proliferation by state i via competition of state j
                sprintf(fname, "numKill_Pheno0_State%d_State%d_Comp.data",
                        i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[i*ca->nStates+j],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                if (i == NC || i == MNC) continue;
                // Differentiation by state i via competition of state j
                sprintf(fname, "numKill_Pheno3_State%d_State%d_Comp.data",
                        i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-2)*ca->nStates+j+70],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                // Kill by state i via competition of state j
                sprintf(fname, "numKill_State%d_State%d_Comp.data",
                        i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-2)*ca->nStates+j+119],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                // num kill state j by state i overall
                sprintf(fname, "numKill_State%d_State%d.data", i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-2)*ca->nStates+j+154],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                if (i == SC || i == MSC) continue;
                // Proliferation by state i via chance of state j
                sprintf(fname, "numKill_Pheno0_State%d_State%d_Chance.data",
                        i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-4)*ca->nStates+j+42],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                // Proliferation by state i overall of state j (competition + chance)
                sprintf(fname, "numKill_Pheno0_State%d_State%d.data", i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-4)*ca->nStates+j+56],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                // Movement by state i via chance of state j
                sprintf(fname, "numKill_Pheno1_State%d_State%d_Chance.data",
                        i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-4)*ca->nStates+j+105],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                // Num kill by state i via chance overall of state j
                sprintf(fname, "numKill_State%d_State%d_Chance.data", i, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[(i-4)*ca->nStates+j+140],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                if (i == TC) continue;
                // Differentiation by CSC via chance of state j
                sprintf(fname, "numKill_Pheno3_State%d_State%d_Chance.data",
                        CSC, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[j+91],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
                // Differentiation by CSC overall of state j
                sprintf(fname, "numKill_Pheno3_State%d_State%d.data", CSC, j);
                save_count_data(fname, ca->headerCount, ticks,
                                countKills[j+98],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
            }
            // Proliferation by competition
            totals[0] += countKills[i*ca->nStates+EMPTY];
            if (i == CSC || i == TC) {
                // Proliferation by chance
                totals[1] += countKills[(i-4)*ca->nStates+EMPTY+42];
            }
            if (i == NC || i == MNC || i == TC) continue;
            // Differenation by competition
            totals[3] += countKills[(i-2)*ca->nStates+EMPTY+70];
        }
        // Proliferation overall
        totals[2] = totals[0] + totals[1];
        // Differentiation overall
        totals[4] = totals[3] + countKills[EMPTY+91];
        // Movement overall
        totals[5] = countKills[(CSC-4)*ca->nStates+EMPTY+105]
                  + countKills[(TC-4)*ca->nStates+EMPTY+105];
        // Competition overall
        totals[6] = totals[2] + totals[3];
        // Chance overall
        totals[7] = countKills[(CSC-4)*ca->nStates+EMPTY+140]
                  + countKills[(TC-4)*ca->nStates+EMPTY+140];
        // Kills overall
        totals[8] = totals[6] + totals[7];
        strcat(totalFname[0], "numKill_Pheno0_Comp.data");
        strcat(totalFname[1], "numKill_Pheno0_Chance.data");
        strcat(totalFname[2], "numKill_Pheno0.data");
        strcat(totalFname[3], "numKill_Pheno3_Comp.data");
        strcat(totalFname[4], "numKill_Pheno3.data");
        strcat(totalFname[5], "numKill_Pheno1.data");
        strcat(totalFname[6], "numKill_Comp.data");
        strcat(totalFname[7], "numKill_Chance.data");
        strcat(totalFname[8], "numKill.data");
        for (i = 0; i < 9; i++)
            save_count_data(totalFname[i], ca->headerCount, ticks,
                            totals[i], 0, 0, 0);

        CudaSafeCall(cudaFree(countData)); countData = NULL;
        CudaSafeCall(cudaFree(countKills)); countKills = NULL;
        CudaSafeCall(cudaFree(countTAC)); countTAC = NULL;
        free(totals); totals = NULL;
        for (i = 0; i < 9; i++) free(totalFname[i]);
        free(totalFname);
    // Save genotypic, phenotypic, and lineage data
    } else if (updateType == 2) {
        CudaSafeCall(cudaMallocManaged((void**)&countDataTemp,
                                       (ca->nStates+ca->nStates*ca->nGenes+4)
                                       *unsgn));
        memset(countDataTemp, 0,
               (ca->nStates+ca->nStates*ca->nGenes+4)*unsgn);
        CudaSafeCall(cudaMallocManaged((void**)&phenoSum,
                                       4*ca->nStates*dbl));
        memset(phenoSum, 0, 4*ca->nStates*dbl);
        CudaSafeCall(cudaMallocManaged((void**)&geneExprSum,
                                       ca->nGenes*ca->nStates*dbl));
        memset(geneExprSum, 0, ca->nGenes*ca->nStates*dbl);
        CudaSafeCall(cudaMallocManaged((void**)&nPosMutSum,
                                       ca->nStates*dbl));
        memset(nPosMutSum, 0, ca->nStates*dbl);
        CudaSafeCall(cudaMallocManaged((void**)&fitnessSum,
                                       ca->nStates*dbl));
        memset(fitnessSum, 0, ca->nStates*dbl);
        collect_data<<< blocks, threads, 0, *stream >>>(
            ca->newGrid, countDataTemp, phenoSum, geneExprSum, nPosMutSum,
            fitnessSum, ca->gridSize, ca->nGenes, ca->nStates
        );
        CudaCheckError();
        CudaSafeCall(cudaStreamSynchronize(*stream));
        save_count_data("numLineages_Mutated.data", ca->headerCount,
                        ticks, ca->nLineage[7], 0, 0, 0);
        save_count_data("numLineages_Non-Mutated.data", ca->headerCount,
                        ticks, ca->nLineage[8], 0, 0, 0);
        for (i = 0; i < ca->nStates; i++) {
            denom = countDataTemp[i];
            if (i == EMPTY) denom = ca->gridSize * ca->gridSize - countDataTemp[EMPTY];
            denom1 = countDataTemp[MNC] + countDataTemp[MSC] + countDataTemp[TC];
            denom2 = countDataTemp[NC] + countDataTemp[SC];

            save_count_data(ca->countFiles[i], ca->headerCount, ticks,
                            countDataTemp[i], ca->stateColors[i].x,
                            ca->stateColors[i].y, ca->stateColors[i].z);
            sprintf(fname, "numLineages_State%d.data", i);
            save_count_data(fname, ca->headerCount, ticks, ca->nLineage[i],
                            ca->stateColors[i].x, ca->stateColors[i].y,
                            ca->stateColors[i].z);

            sprintf(fname, "fitness_State%d", i);
            update_avg_data(i, fname, ticks, (denom == 0) ? 0 : fitnessSum[i] / denom,
                            (denom1 == 0) ? 0 : (fitnessSum[i] - fitnessSum[NC] - fitnessSum[SC]) / denom1,
                            (denom2 == 0) ? 0 : (fitnessSum[NC] + fitnessSum[SC]) / denom2,
                            dim3(ca->stateColors[i].x, ca->stateColors[i].y, ca->stateColors[i].z));

            sprintf(fname, "numPosMut_State%d", i);
            update_avg_data(i, fname, ticks, (denom == 0) ? 0 : nPosMutSum[i] / denom,
                            (denom1 == 0) ? 0 : (nPosMutSum[i] - nPosMutSum[NC] - nPosMutSum[SC]) / denom1,
                            (denom2 == 0) ? 0 : (nPosMutSum[NC] + nPosMutSum[SC]) / denom2,
                            dim3(ca->stateColors[i].x, ca->stateColors[i].y, ca->stateColors[i].z));
            for (j = 0; j < ca->nGenes; j++) {
                sprintf(fname, "geneExpr%d_State%d", j, i);
                update_avg_data(i, fname, ticks, (denom == 0) ? 0 : geneExprSum[i*ca->nGenes+j] / denom,
                                (denom1 == 0) ? 0 : (geneExprSum[i*ca->nGenes+j] - geneExprSum[NC*ca->nGenes+j]
                                                   - geneExprSum[SC*ca->nGenes+j]) / denom1,
                                (denom2 == 0) ? 0 : (geneExprSum[NC*ca->nGenes+j] + geneExprSum[SC*ca->nGenes+j]) / denom2,
                                dim3(ca->stateColors[i].x, ca->stateColors[i].y, ca->stateColors[i].z));
                if (i == EMPTY) {
                    save_count_data(ca->countFiles[j+15], ca->headerCount, ticks,
                                    countDataTemp[j+ca->nStates+4], ca->geneColors[j].x,
                                    ca->geneColors[j].y, ca->geneColors[j].z);
                    continue;
                }
                save_count_data(ca->countFiles[i*ca->nGenes+(3*ca->nStates+ca->nGenes+15)+j],
                                ca->headerCount, ticks,
                                countDataTemp[i*ca->nGenes+(ca->nGenes+ca->nStates)+j],
                                ca->stateColors[i].x, ca->stateColors[i].y,
                                ca->stateColors[i].z);
            }

            for (j = 0; j < 4; j++) {
                if (j == DIFF && (i == NC || i == MNC || i == TC)) continue;
                numerator1 = phenoSum[i*4+j] - phenoSum[NC*4+j] - phenoSum[SC*4+j];
                numerator2 = phenoSum[NC*4+j] + phenoSum[SC*4+j];
                if (i == EMPTY && j == DIFF) {
                    denom -= (countDataTemp[NC] + countDataTemp[MNC]
                            + countDataTemp[TC]);
                    numerator1 -= (phenoSum[MNC*4+j] + phenoSum[TC*4+j]);
                    denom1 -= (countDataTemp[MNC] + countDataTemp[TC]);
                    numerator2 -= phenoSum[NC*4+j];
                    denom2 -= countDataTemp[NC];
                }

                sprintf(fname, "chancePheno%d_State%d", j, i);
                update_avg_data(i, fname, ticks, (denom == 0) ? 0 : phenoSum[i*4+j] / denom,
                                (denom1 == 0) ? 0 : numerator1 / denom1,
                                (denom2 == 0) ? 0 : numerator2 / denom2,
                                dim3(ca->stateColors[i].x, ca->stateColors[i].y, ca->stateColors[i].z));
            }

            if (i == SC || i == MSC || i == CSC) continue;
            step = 0;
            if (i == TC) step = 3;
            else if (i != EMPTY) step = i + 1;
            save_count_data(ca->countFiles[ca->nStates+step],
                            ca->headerCount, ticks,
                            countDataTemp[ca->nStates+step],
                            ca->stateColors[i].x, ca->stateColors[i].y,
                            ca->stateColors[i].z);
        }

        CudaSafeCall(cudaFree(phenoSum)); phenoSum = NULL;
        CudaSafeCall(cudaFree(geneExprSum)); geneExprSum = NULL;
        CudaSafeCall(cudaFree(fitnessSum)); fitnessSum = NULL;

        return countDataTemp;
    // Carcinogen Function data
    } else if (updateType == 3) {
        if (ca->nCarcin != 0) {
            total = 0;
            for (i = 0; i < ca->maxNCarcin; i++)
                if (ca->activeCarcin[i]) {
                    sprintf(fname, "Carcin%d_max.data", ca->carcins[i].carcinIdx);
                    save_count_data(fname, "# t max color\n", ticks, *ca->carcins[i].maxVal, 0, 0, 0);
                    total += *ca->carcins[i].maxVal;
                }
            save_count_data("Carcins_max.data", "# t max color\n", ticks, total, 0, 0, 0);
            save_count_data("Carcins_avgMax.data", "# t avg_max color\n", ticks, total / (double) ca->nCarcin,
                            0, 0, 0);
        }
    }

    return NULL;
}

bool tumour_remover(bool excise, unsigned *radius, unsigned *centerX,
                    unsigned *centerY, unsigned numExcisionLocations, unsigned *numTC,
                    CA *ca, unsigned ticks, dim3 blocks, dim3 threads, cudaStream_t *streams)
{
    unsigned *rTC, i;
    int *tcX, *tcY;
    bool excisionPerformed = false;

    if (ca->exciseCount <= ca->maxExcise
        && (excise || (ca->maxTTCAlive != -1
        && ca->timeTCAlive != 0 && ca->timeTCAlive % ca->maxTTCAlive == 0)
        || (ca->perfectExcision && !ca->exciseCount && ca->excisionTime+1 == ca->timeTCAlive))) {
        if (ca->exciseCount == ca->maxExcise) {
            printf("The maximum number of excision have been performed.\n");
            return excisionPerformed;
        }
        if (!ca->perfectExcision) {
            for (i = 0; i < numExcisionLocations; i++) {
                if (ca->exciseCount == ca->maxExcise) {
                    printf("The maximum number of excisions have been performed.\n");
                    break;
                }
                CudaSafeCall(cudaMallocManaged((void**)&rTC,
                                               sizeof(unsigned)));
                *rTC = 0;
                CudaSafeCall(cudaMallocManaged((void**)&tcX, sizeof(int)));
                CudaSafeCall(cudaMallocManaged((void**)&tcY, sizeof(int)));
                *tcX = -1; *tcY = -1;
                if (radius[i] == 0 && *numTC != 0) {
                    set_excision_circle<<< blocks, threads, 0, streams[0] >>>(
                        ca->newGrid, ca->gridSize, rTC, tcX, tcY
                    );
                    CudaCheckError();
                    CudaSafeCall(cudaStreamSynchronize(streams[0]));
                } else if (radius[i] != 0) {
                    *rTC = radius[i]; *tcX = centerX[i]; *tcY = centerY[i];
                }
                if (*rTC != 0) {
                    excision<<< blocks, threads, 0, streams[0] >>>(
                        ca->newGrid, ca->gridSize, ca->nGenes,
                        *rTC, *tcX, *tcY, numTC
                    );
                    CudaCheckError();
                    CudaSafeCall(cudaStreamSynchronize(streams[0]));
                    printf("Radius of excision: %d, center: (%d, %d)\n",
                           *rTC, *tcX, *tcY);

                    excisionPerformed = true;
                    ca->radius[ca->exciseCount+1] = *rTC;
                    ca->centerX[ca->exciseCount+1] = *tcX;
                    ca->centerY[ca->exciseCount+1] = *tcY;
                    if (*numTC == 0) ca->timeTCAlive = 0;
                    ca->exciseCount++;
                }
                CudaSafeCall(cudaFree(rTC)); rTC = NULL;
                CudaSafeCall(cudaFree(tcX)); tcX = NULL;
                CudaSafeCall(cudaFree(tcY)); tcY = NULL;
            }
        } else if (ca->perfectExcision && ca->tcFormed[ca->exciseCount]) {
            tumour_excision<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->removeField, ca->gridSize, ca->nGenes,
                ca->maxNeighDepth
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            excisionPerformed = true;
            ca->timeTCAlive = 0;
            ca->exciseCount++;
        }

        if (excisionPerformed)
            printf("Excision was performed at time step %d.\n",
                   ticks);
    }

    return excisionPerformed;
}

void anim_gpu_ca(uchar4* outputBitmap, unsigned dim, CA *ca,
                 unsigned ticks, bool display, bool *paused,
                 bool excise, unsigned *radius, unsigned *centerX, unsigned *centerY,
                 unsigned numExcisionLocations, bool *earlyStop, bool *keys)
{
    size_t unsgn = sizeof(unsigned);
    unsigned i, *numTC, *countData, *countKills, *countTAC,
             numCells, prevExciseCount = ca->exciseCount;
    bool firstTCFormedCheck[prevExciseCount+1];
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize)),
         threads(ca->blockSize, ca->blockSize);
    cudaStream_t streams[ca->maxNCarcin+2], *streamsExcise;

    update_params(ca, keys);

    for (i = 0; i < prevExciseCount+1; i++)
        firstTCFormedCheck[i] = ca->tcFormed[i];

    for (i = 0; i < ca->maxNCarcin+2; i++)
        CudaSafeCall(cudaStreamCreate(&streams[i]));

    if (ticks <= ca->maxT) {
        if (ticks == 0) {
            if (display || ca->save) {
                display_ca<<< blocks, threads, 0, streams[0] >>>(
                    outputBitmap, ca->newGrid, ca->gridSize,
                    ca->cellSize, dim, ca->stateColors
                );
                CudaCheckError();
            }
            if (!(*paused))
                save_cell_data_to_file(ca, ticks, blocks, threads, &streams[1]);
            CudaSafeCall(cudaStreamSynchronize(streams[0]));
            CudaSafeCall(cudaStreamSynchronize(streams[1]));
        } else if (!(*paused)) {
            mutate_grid<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->gridSize, ca->NN,
                ca->carcins, ca->activeCarcin, ticks
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            if (ca->nCarcin != 0)
                for (i = 0; i < ca->maxNCarcin; i++)
                    if (ca->activeCarcin[i])
                        ca->carcins[i].time_step(ca->blockSize, &streams[i+2]);

            cells_gpu_to_gpu_cpy<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            // Do this before rule possibly since after this function is applied
            // it could change an empty cell into a new state or a cell that no
            // longer is of a certain state into a state
            CudaSafeCall(cudaMallocManaged((void**)&countData,
                                           35*unsgn));
            memset(countData, 0, 35*unsgn);
            update_states<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize,
                ca->nGenes, ticks, countData
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            update_graph_data(ca, ticks, 0, countData);
            CudaSafeCall(cudaFree(countData)); countData = NULL;

            CudaSafeCall(cudaMallocManaged((void**)&countData,
                                           (3 * ca->nStates + 4)*unsgn));
            memset(countData, 0, (3 * ca->nStates + 4)*unsgn);
            CudaSafeCall(cudaMallocManaged((void**)&countKills,
                                           (26 * ca->nStates)*unsgn));
            memset(countKills, 0, (26 * ca->nStates)*unsgn);
            CudaSafeCall(cudaMallocManaged((void**)&countTAC,
                                           9*unsgn));
            memset(countTAC, 0, 9*unsgn);
            rule<<< blocks, threads, 0, streams[0] >>>(
                ca->newGrid, ca->prevGrid, ca->gridSize,
                ca->nGenes, ticks, countData, countKills, countTAC
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            update_graph_data(ca, ticks, 1, countData, countKills, countTAC);

            CudaSafeCall(cudaMallocManaged((void**)&numTC,
                                           2*sizeof(unsigned)));
            numTC[0] = 0; numTC[1] = 0;
            streamsExcise = (cudaStream_t*)malloc((ca->exciseCount+1)*sizeof(cudaStream_t));
            for (i = 0; i < ca->exciseCount+1; i++) {
                CudaSafeCall(cudaStreamCreate(&streamsExcise[i]));
                if (i != 0 && ca->tcFormed[i]) continue;
                check_CSC_or_TC_formed<<< blocks, threads, 0, streamsExcise[i] >>>(
                    ca->newGrid, ca->prevGrid, ca->gridSize, ticks, ca->cscFormed,
                    ca->tcFormed, i, ca->timeTCDead, ca->perfectExcision,
                    ca->radius, ca->centerX, ca->centerY, numTC
                );
                CudaCheckError();
            }

            for (i = 0; i < ca->exciseCount+1; i++) {
                CudaSafeCall(cudaStreamSynchronize(streamsExcise[i]));
                CudaSafeCall(cudaStreamDestroy(streamsExcise[i]));
            }
            free(streamsExcise); streamsExcise = NULL;

            if (numTC[0] != 0 && numTC[1] != 0) ca->timeTCAlive++;
            else ca->timeTCAlive = 0;

            for (i = 0; i < ca->exciseCount+1; i++)
                if (!ca->tcFormed[i])
                    ca->timeTCDead[i]++;

            tumour_remover(excise, radius, centerX, centerY, numExcisionLocations,
                           numTC, ca, ticks, blocks, threads, streams);
            CudaSafeCall(cudaFree(numTC)); numTC = NULL;

            memset(ca->nLineage, 0, (ca->nStates+2)*unsgn); *ca->nLineageCells = 0;
            memset(ca->cellLineage, 0, ca->gridSize*ca->gridSize*unsgn);
            memset(ca->stateInLineage, 0,
                   (ca->nStates-1)*ca->gridSize*ca->gridSize*sizeof(bool));
            update_lineage_data<<< blocks, threads, 0, streams[1] >>>(
                ca->newGrid, ca->cellLineage, ca->stateInLineage,
                ca->nLineage, ca->nLineageCells, ca->gridSize
            );
            CudaCheckError();
            update_lineage_data<<< blocks, threads, 0, streams[1] >>>(
                ca->newGrid, ca->cellLineage, ca->stateInLineage,
                ca->nLineage, ca->nLineageCells, ca->gridSize, true
            );
            CudaCheckError();

            save_cell_data_to_file(ca, ticks, blocks, threads, &streams[0]);
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            reset_rule_params<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize
            );
            CudaCheckError();
            CudaSafeCall(cudaStreamSynchronize(streams[0]));

            cells_gpu_to_gpu_cpy<<< blocks, threads, 0, streams[0] >>>(
                ca->prevGrid, ca->newGrid, ca->gridSize, ca->nGenes
            );
            CudaCheckError();

            if ((display || ca->save)
             && ticks % ca->framerate == 0) {
                display_ca<<< blocks, threads, 0, streams[0] >>>(
                    outputBitmap, ca->newGrid, ca->gridSize,
                    ca->cellSize, dim, ca->stateColors);
                CudaCheckError();
            }
        }

        if (!(*paused)) {
            countData = update_graph_data(ca, ticks, 2, NULL, NULL, NULL, &streams[1]);

            numCells = ca->gridSize * ca->gridSize;
            if (countData[EMPTY] == numCells
             || countData[CSC] + countData[TC] + countData[EMPTY] == numCells)
                *earlyStop = true;

            CudaSafeCall(cudaFree(countData)); countData = NULL;
        }

        for (i = 0; i < ca->maxNCarcin+2; i++)
            CudaSafeCall(cudaStreamSynchronize(streams[i]));

        if (!(*paused)) {
            update_graph_data(ca, ticks, 3);
            if (ca->nCarcin > 0)
                for (i = 0; i < ca->maxNCarcin; i++)
                    if (ca->activeCarcin[i]) ca->carcins[i].t++;

            if (ca->save && ticks % ca->framerate == 0)
                save_image(outputBitmap, dim, ca->blockSize, NULL,
                           ticks, ca->devId2);
        }
    }

    for (i = 0; i < ca->maxNCarcin+2; i++)
        CudaSafeCall(cudaStreamDestroy(streams[i]));

    // First TC formed so pause so user can do something about it
    if (ca->pauseOnFirstTC && display)
        for (i = 0; i < prevExciseCount+1; i++)
            if (firstTCFormedCheck[i] != ca->tcFormed[i]) {
                *paused = true;
                if (i == 0)
                    printf("Simulation was paused due to the first TC forming.\n");
                else
                    printf("Simulation was paused due to the first TC forming after an excision.\n");
                break;
            }

    if (ca->excisionTime == ca->timeTCAlive && !ca->perfectExcision
     && ca->exciseCount == 0 && display) {
        *paused = true;
        printf("An excision can be performed as the TC have been alive for %d time steps.\n",
               ca->excisionTime);
    }
}

void anim_gpu_lineage(uchar4* outputBitmap, unsigned dim, CA *ca, unsigned idx,
                      unsigned stateIdx, unsigned ticks, bool display, bool paused)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);
    unsigned i;
    char fname[15] = { '\0' };

    if (idx == 2)
        for (i = 0; i < 20; i++) {
            ca->maxLineages[i] = 0;
            max_lineage<<< blocks, threads >>>(
                ca->newGrid, ca->cellLineage, ca->maxLineages,
                i, stateIdx, ca->gridSize
            );
            CudaCheckError();
            CudaSafeCall(cudaDeviceSynchronize());
        }

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        if (idx == 0)
            display_genes<<< blocks, threads >>>(
                outputBitmap, ca->newGrid, ca->gridSize, ca->cellSize, dim,
                ca->NN->nOut, ca->geneColors
            );
        else if (idx == 2)
            display_max_lineages<<< blocks, threads >>>(
                outputBitmap, ca->newGrid, ca->cellLineage, ca->stateInLineage,
                ca->maxLineages, stateIdx, ca->gridSize, ca->cellSize, dim,
                ca->lineageColors
            );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    if (idx == 1) {
        memset(ca->percentageCounts, 0, 10*sizeof(unsigned));
        display_lineage_heatmap<<< blocks, threads >>>(
            outputBitmap, ca->newGrid, ca->cellLineage,
            ca->nLineageCells, ca->percentageCounts,
            ca->gridSize, ca->cellSize, dim, ca->heatmap
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
        if (!paused)
            for (i = 0; i < 10; i++) {
                sprintf(fname, "%d-%d%%.data",
                        i == 0 ? i * 10 : i * 10 + 1, (i + 1) * 10);
                save_count_data(fname, "# t num_lineages color\n", ticks,
                                ca->percentageCounts[i], ca->heatmap[i].x,
                                ca->heatmap[i].y, ca->heatmap[i].z);
            }
    }

    if (ca->save && ticks <= ca->maxT && !paused
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[idx == 2 ? idx+stateIdx+1 : idx+1],
                   ticks, ca->devId2);
}

void anim_gpu_carcin(uchar4* outputBitmap, unsigned dim, CA *ca,
                     unsigned carcinIdx, unsigned ticks,
                     bool display, bool paused)
{
    dim3 blocks(NBLOCKS(ca->gridSize, ca->blockSize),
                NBLOCKS(ca->gridSize, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_carcin<<< blocks, threads >>>(
            outputBitmap, &ca->carcins[carcinIdx],
            ca->gridSize, ca->cellSize, dim
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    if (ca->save && ticks <= ca->maxT && !paused
     && ticks % ca->framerate == 0)
        save_image(outputBitmap, dim, ca->blockSize,
                   ca->prefixes[carcinIdx+10], ticks, ca->devId2);

}

void anim_gpu_cell(uchar4* outputBitmap, unsigned dim, CA *ca,
                   unsigned cellIdx, unsigned ticks, bool display)
{
    dim3 blocks(NBLOCKS(dim, ca->blockSize), NBLOCKS(dim, ca->blockSize));
    dim3 threads(ca->blockSize, ca->blockSize);

    if ((display || ca->save)
     && ticks % ca->framerate == 0) {
        display_cell_data<<< blocks, threads >>>(
            outputBitmap, ca->newGrid, cellIdx, dim, ca->NN->nOut,
            ca->stateColors
        );
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }
}

void anim_gpu_timer_and_saver(CA *ca, bool start, unsigned ticks, bool paused,
                              bool windowsShouldClose, bool earlyStop)
{
    unsigned i, videoFramerate = 24, nVideos = ca->maxNCarcin + ca->nStates + 3;
    long int startPoint = (int) (ticks - videoFramerate);
    char *vidListName= NULL; struct stat buffer;
    double timerVideos;

    if (start && !(windowsShouldClose || earlyStop)) {
        ca->startStep = clock();
        printf("starting %d\n", ticks);
    } else if (!start) {
        ca->endStep = clock();
        printf("%d took %f seconds to complete.\n", ticks,
               (double) (ca->endStep - ca->startStep) / CLOCKS_PER_SEC);
    }
    if (ticks == 0 && !paused) ca->start = clock();

    if (!start && ca->save && (windowsShouldClose || (!paused && ((ticks != 0
     && startPoint % videoFramerate == 0) || ticks == ca->maxT || earlyStop)))) {
        if (ticks < videoFramerate) {
            startPoint = 0;
            videoFramerate = 1;
        }
        if (ticks == ca->maxT && ticks > videoFramerate)
            startPoint = (int) (ticks - ticks % videoFramerate);
        timerVideos = omp_get_wtime();
        #pragma omp parallel for num_threads(8) schedule(static, 2)\
                default(shared) private(i)
        for (i = 0; i < nVideos; i++) {
            if (i >= nVideos - ca->maxNCarcin && !ca->activeCarcin[i-10])
                continue;
            fflush(stdout);
            printf("Saving video %s.\n", ca->outNames[i]);
            save_video(ca->prefixes[i], ca->outNames[i],
                       startPoint, videoFramerate);
            fflush(stdout);
            printf("Finished video %s.\n", ca->outNames[i]);
        }
        printf("It took %f seconds to finish updating the videos.\n",
               omp_get_wtime() - timerVideos);
    }

    if (!start && (ticks == ca->maxT || windowsShouldClose || earlyStop)) {
        ca->end = clock();
        printf("It took %f seconds to run the %d time steps.\n",
               (double) (ca->end - ca->start) / CLOCKS_PER_SEC, ticks);
        if (!ca->save) return;
        for (i = 0; i < nVideos; i++) {
            vidListName = (char*)calloc((i == 0 ? 0
                                         : strlen(ca->prefixes[i])) + 11, 1);
            if (i != 0) strcat(vidListName, ca->prefixes[i]);
            strcat(vidListName, "videos.txt");
            if (stat(vidListName, &buffer) == -1) continue;
            if (remove(vidListName) != 0)
                fprintf(stderr, "Error removing %s.\n", vidListName);
            free(vidListName); vidListName = NULL;
        }
    }
}

#endif // __ANIM_FUNCTIONS_H__
