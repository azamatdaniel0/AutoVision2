const config = require('config'),
      fs = require('fs'),
      path = require('path'),
      jsonStore = require('../../app/helpers/jsonStore'),
      checkDir = require('../../app/helpers/checkDir'),
      checkDirStructure = require('../../app/helpers/checkDirStructure'),
      moveDatasetFiles = require('../../app/helpers/moveDatasetFiles'),
      joinAnnotationOcrDataset = require('../../app/helpers/joinAnnotationOcrDataset'),
      arrayShuffle = require('../../app/helpers/arrayShuffle'),
      //statsTools = require('../../app/helpers/statsTools'),
      sizeOf = require('image-size'),
      md5File = require('md5-file')
;

async function moveBody (options, filterFunction=function (data, data_anb) { return true }, splitRate=1) {
        const sourceDir = options.srcDir || './draft',
              targetDir = options.targetDir || './checked',
              testMode = options.test || false,
              annExt = '.' + config.dataset.ann.ext,
              grouping = options.grouping || 0,
              src= { annPath: path.join(sourceDir, config.dataset.ann.dir),
                                     anbPath: path.join(sourceDir, config.dataset.anb.dir) }
        ;
        let checkedAnn = [],
            checkedImg = [],
            checkedAnb = [],
            checkedBox = [],
            checkedSrc = [],
            stayAnbs = [],
            stayBoxs = [],
            staySrcs = [],
            itemsStaySrcIdx = {},
            itemsStay = []
        ;
        checkDirStructure(sourceDir,[config.dataset.anb.dir,config.dataset.ann.dir,
                          config.dataset.img.dir,config.dataset.box.dir,config.dataset.src.dir], true);
        checkDirStructure(targetDir, [config.dataset.anb.dir,config.dataset.ann.dir,
                          config.dataset.img.dir,config.dataset.box.dir,config.dataset.src.dir], true);

        fs.readdir(src.annPath, async function(err, items) {
                if (splitRate != 1) {
                    let sItems = arrayShuffle(items),
                        cnt = (splitRate>1)?Math.round(splitRate):Math.round(sItems.length * splitRate);
                    items = sItems.slice(0,cnt);
                    itemsStay = sItems.slice(cnt);
                }
                if (grouping == 1) {
                    for (let filename of itemsStay) {
                        const fileObj = path.parse(filename);
                            if (fileObj.ext === annExt) {
                                const
                                    anbName = `${fileObj.name.split('-')[0]}.${config.dataset.anb.ext}`,
                                    anbFileName = path.join( src.anbPath, anbName),
                                    data_anb = require(path.isAbsolute(anbFileName) ? anbFileName : path
                                        .join(process.cwd(), anbFileName)),
                                    srcName = data_anb.src
                                ;
                                itemsStaySrcIdx[srcName] = 1
                            }
                    }
                }
                for (let filename of items) {
                        const fileObj = path.parse(filename);
                        if (fileObj.ext === annExt) {
                                const
                                    annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                                    annFileName = path.join( src.annPath, annName),
                                    data = require(path.isAbsolute(annFileName)?annFileName:path
                                                   .join(process.cwd(), annFileName)),
                                    anbName = `${fileObj.name.split('-')[0]}.${config.dataset.anb.ext}`,
                                    anbFileName = path.join( src.anbPath, anbName),
                                    data_anb = require(path.isAbsolute(anbFileName)?anbFileName:path
                                                   .join(process.cwd(), anbFileName)),
                                    boxName = `${fileObj.name.replace(/-line-\d/,'')}.${config.dataset.box.ext}`,
                                    srcName = data_anb.src,
                                    imgName = `${data.name}.${config.dataset.img.ext}`
                                ;
                                // console.log(`data`);
                                // console.log(data)
                                if (filterFunction(data, data_anb) && itemsStaySrcIdx[srcName] == undefined) {
                                        checkedAnn.push(annName);
                                        checkedImg.push(imgName);
                                        checkedAnb.push(anbName);
                                        checkedBox.push(boxName);
                                        checkedSrc.push(srcName);
                                } else {
                                    stayAnbs.push(anbName);
                                    stayBoxs.push(boxName);
                                    staySrcs.push(srcName);
                                }
                        }
                }
                console.log(`Checked: ${checkedAnn.length}`);
                moveDatasetFiles({sourceDir, targetDir,
                    Anbs: checkedAnb, anbDir:config.dataset.anb.dir, stayAnbs,
                    Anns: checkedAnn, annDir:config.dataset.ann.dir,
                    Imgs: checkedImg, imgDir:config.dataset.img.dir,
                    Boxs: checkedBox, boxDir:config.dataset.box.dir, stayBoxs,
                    Srcs: checkedSrc, srcDir:config.dataset.src.dir, staySrcs,
                    test:testMode});
        });
}



async function moveAllItemPack (sourceDir, targetDir, baseName, removeDiffRegions = false, debug = false){
    const
        anbDirSrc = path.join(sourceDir, config.dataset.anb.dir),
        annDirSrc = path.join(sourceDir, config.dataset.ann.dir),
        boxDirSrc = path.join(sourceDir, config.dataset.box.dir),
        imgDirSrc = path.join(sourceDir, config.dataset.img.dir),
        srcDirSrc = path.join(sourceDir, config.dataset.src.dir),
        anbDirTarget = path.join(targetDir, config.dataset.anb.dir),
        annDirTarget = path.join(targetDir, config.dataset.ann.dir),
        boxDirTarget = path.join(targetDir, config.dataset.box.dir),
        imgDirTarget = path.join(targetDir, config.dataset.img.dir),
        srcDirTarget = path.join(targetDir, config.dataset.src.dir),

        anbPathSrc = path.join(anbDirSrc, `${baseName}.${config.dataset.anb.ext}`),
        anbPathTarget = path.join(anbDirTarget, `${baseName}.${config.dataset.anb.ext}`),
        anbDataSrc = require(anbPathSrc),
        srcPathSrc = path.join(srcDirSrc, anbDataSrc.src),
        srcPathTarget = path.join(srcDirTarget, anbDataSrc.src)
    ;

    // Remove old regions
    if (removeDiffRegions && fs.existsSync(anbPathTarget)) {
        if (debug) { console.log(`  moveAllItemPack: try to remove old items files for ${baseName}`) }
        const
            anbDataTarget = require(anbPathTarget),
            remove_regions = {}
        ;
        for (const region_key in anbDataTarget.regions) {
            if (anbDataSrc.regions[region_key] == undefined) {
                remove_regions[region_key] = anbDataTarget.regions[region_key]
            }
        }
        if (debug) { console.log(`  moveAllItemPack:   found ${Object.keys(remove_regions).length} old regions: ${Object.keys(remove_regions)}`) }
        for (const region_key in remove_regions) {
            const
                region = remove_regions[region_key],
                boxPathTarget = path.join(boxDirTarget, `${region_key}.${config.dataset.box.ext}`)
            ;
            // Remove box
            if (fs.existsSync(boxPathTarget)) {
                if (debug) { console.log(`  moveAllItemPack:   delete box file ${boxPathTarget}`) }
                fs.rmSync(boxPathTarget);
            }

            for (const line_key in region.lines) {
                const
                    annPathTarget = path.join(annDirTarget, `${region_key}-line-${line_key}.${config.dataset.ann.ext}`),
                    imgPathTarget = path.join(imgDirTarget, `${region_key}-line-${line_key}.${config.dataset.img.ext}`)
                ;

                // Remove line
                if (fs.existsSync(annPathTarget)) {
                    if (debug) { console.log(`  moveAllItemPack:   delete ann file ${annPathTarget}`) }
                    fs.rmSync(annPathTarget);
                }
                if (fs.existsSync(imgPathTarget)) {
                    if (debug) { console.log(`  moveAllItemPack:   delete img file ${annPathTarget}`) }
                    fs.rmSync(imgPathTarget);
                }
            }
        }
    }

    // Move anb
    if (fs.existsSync(anbPathTarget)) {
        fs.rmSync(anbPathTarget);
    }
    fs.renameSync(anbPathSrc, anbPathTarget);

    // Move src
    if (fs.existsSync(srcPathSrc)) {
        fs.renameSync(srcPathSrc, srcPathTarget);
    }

    // Move regions files
    if (debug) { console.log(`  moveAllItemPack: try to move ${Object.keys(anbDataSrc.regions).length} regions: ${Object.keys(anbDataSrc.regions)}`) }
    for (const region_key in anbDataSrc.regions) {
        const
            region = anbDataSrc.regions[region_key],
            boxPathSrc = path.join(boxDirSrc, `${region_key}.${config.dataset.box.ext}`),
            boxPathTarget = path.join(boxDirTarget, `${region_key}.${config.dataset.box.ext}`)
        ;
        // Move box
        if (fs.existsSync(boxPathSrc)) {
            fs.renameSync(boxPathSrc, boxPathTarget)
        }
        for (const line_key in region.lines) {
            const
                annPathSrc = path.join(annDirSrc, `${region_key}-line-${line_key}.${config.dataset.ann.ext}`),
                annPathTarget = path.join(annDirTarget, `${region_key}-line-${line_key}.${config.dataset.ann.ext}`),
                imgPathSrc = path.join(imgDirSrc, `${region_key}-line-${line_key}.${config.dataset.img.ext}`),
                imgPathTarget = path.join(imgDirTarget, `${region_key}-line-${line_key}.${config.dataset.img.ext}`)
            ;
            if (debug) { console.log(`  moveAllItemPack:   checking ann line file: ${annPathTarget}`) }
            if (fs.existsSync(annPathSrc)) {
                if (fs.existsSync(annPathTarget)) {
                    if (debug) { console.log(`  moveAllItemPack:   remove old ann-file ${annPathTarget}`) }
                    fs.rmSync(annPathTarget);
                }
                if (fs.existsSync(imgPathTarget)) {
                    if (debug) { console.log(`  moveAllItemPack:   remove old img-file ${annPathTarget}`) }
                    fs.rmSync(imgPathTarget);
                }
                // Move ann
                if (debug) { console.log(`  moveAllItemPack:   rename ann-file to ${annPathTarget}`) }
                fs.renameSync(annPathSrc, annPathTarget);

                // Move img
                if (debug) { console.log(`  moveAllItemPack:   rename img-file to ${imgPathTarget}`) }
                fs.renameSync(imgPathSrc, imgPathTarget);
            }
        }
    }
}




/**
 * EN: Console script example
 * UA: Приклад консольного скрипту
 * RU: Пример консольного скрипта
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=index
 * @module controllers/defaultController
 */
async function index (options) {
    console.log('Hello world defaultController & index action with options: ' + JSON.stringify(options));
}


/**
 * EN: Creating an OCR dataset Nomeroff Net
 * UA: Створення OCR-датасету Nomeroff Net
 * RU: Создание OCR-датасета Nomeroff Net
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=createAnnotations  \
 *                                               --opt.baseDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.imageDir=../data/examples/numberplate_zone_images/
 */
async function createAnnotations (options) {
        let baseDir = options.baseDir || config.dataset.baseDir;
        let imageDir = options.imageDir || Error("You must setup imageDir param");
        checkDir(baseDir, true);

        const imgPath = path.join(baseDir, config.dataset.img.dir),
              annPath = path.join(baseDir, config.dataset.ann.dir),
              imgExt = '.'+config.dataset.img.ext,
              tplPath = path.join(config.template.path, config.template.annDefault),
              annTrmplate = require(tplPath);
        checkDir(imgPath, true);
        checkDir(annPath,true);

        console.log("readdir", imageDir);
        fs.readdir(imageDir, async function(err, items) {
                for (let filename of items) {
                        const fileObj = path.parse(filename);
                        if (fileObj.ext === imgExt) {
                                const annFile = path.join(annPath, `${fileObj.name}.${config.dataset.ann.ext}`),
                                      imgFile = path.join(imageDir, filename),
                                      resImgFile = path.join(imgPath, filename),
                                      imgSize = sizeOf(imgFile);
                                fs.copyFileSync(imgFile, resImgFile)
                                let data = Object.assign(annTrmplate,{
                                        description: fileObj.name,
                                        name: fileObj.name,
                                        size: {
                                                width: imgSize.width,
                                                height: imgSize.height
                                        }
                                });
                                console.log(`Store ${annFile}`);
                                await jsonStore(annFile, data);
                        }
                }
        });
}


/**
 * EN: Move the moderated data to a separate folder from the OCR dataset
 * UA: Перенести в окрему папку з OCR-датасету промодейовані дані
 * RU: Перенести в отдельную папку из OCR-датасета промодеированные данные
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveChecked \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/val/
 */
async function moveChecked (options) {
    const
        myFilter = function (data, data_anb) {
            return data.moderation !== undefined
                   && data.moderation.isModerated !== undefined
                   && data.moderation.isModerated
        }
        moveBody(options, myFilter);
}


/**
 * EN: Divide the dataset into 2 parts in a given proportion (move from a given folder to a specified one with a given proportion)
 * UA: Поділити датасет на 2 частини у заданій пропорції (перенести із заданої папки у зазначену із заданою пропорцією)
 * RU: Поделить датасет на 2 части в заданой пропорции (перенести из заданной папки в указанную с заданной пропорцией)
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=dataSplit \
 *                                               --opt.rate=0.5 --opt.grouping=1 \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/val/ \
 *                                               --opt.test=1 #use opt.test=1 if you want emulate split process
 */
async function dataSplit (options) {
        const
            splitRate = options.rate || 0.2,
            myFilter = function (data, data_anb) { return true }
        ;
        moveBody(options, myFilter, splitRate);
}


/**
 * EN: Divide the dataset into 2 parts in a given proportion (move from a given folder to a specified one with a given proportion)
 * UA: Об'єднання 2-х датасетів, в яких є однакові вихідні файли у /anb та /src
 *     Об'єднуємо з srcDir у targetDir. Якщо є два однакових файла /anb/<name>.json, то файл з srcDir перезаписує файл у targetDir,
 *     при цьому якщо в якомусь region є "rebuilded": true, то по цьому регіону будуть перезаписані всі файли в targetDir файлами з srcDir
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=mergeDatasets \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_moderated/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_checked/ \
 *                                               --opt.debug=1
 */
async function mergeDatasets (options) {
        const
            sourceDir = options.srcDir || './moderated',
            targetDir = options.targetDir || './checked',
            debug = Boolean(options.debug || false),
            anbDirSrc = path.join(sourceDir, config.dataset.anb.dir),
            anbDirTarget = path.join(targetDir, config.dataset.anb.dir),
            anbFilesSrc = fs.readdirSync(anbDirSrc)
        ;

        if (debug) { console.log(`Prepare ${anbFilesSrc.length} items for merging`) }
        let cnt= 0, cnt_for_move = 0, cnt_for_merge = 0;
        for (const anbFileSrc of anbFilesSrc) {
            const
                baseName = anbFileSrc.split('.'+config.dataset.anb.ext)[0],
                anbPathSrc = path.join(anbDirSrc, anbFileSrc),
                anbPathTarget = path.join(anbDirTarget, anbFileSrc),
                anbDataSrc = require(anbPathSrc)
            ;
            if (debug) { console.log(`Processing ${baseName}...`) }
            if (fs.existsSync(anbPathTarget)) {
                //if (anbDataSrc.rebuilded != undefined && anbDataSrc.rebuilded) {
                if (debug) { console.log(`  exist in target dataset`) }
                moveAllItemPack(sourceDir,targetDir, baseName, true, debug);
                cnt_for_merge++;
                //}
            } else {
                moveAllItemPack(sourceDir,targetDir, baseName, false, debug);
                cnt_for_move++;
            }
            cnt++;
            if (debug) { if (cnt % 1000) console.log(`${cnt} items done`) }
        }
        if (debug) {
            console.log(`Total: ${cnt} items moved`)
            console.log(`Merging: ${cnt_for_merge} items`)
            console.log(`Moved: ${cnt_for_move} items`)
        }
}



/**
 * EN: Move to a separate folder "garbage" ("region_id": 0) from the OCR dataset
 * UA: Перенести в папку "сміття" ("region_id": 0) з OCR-датасету
 * RU: Перенести в одельную папку "мусор" ("region_id": 0) из OCR-датасета
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveGarbage  \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/garbage/
 */
async function moveGarbage (options) {
    const
        myFilter = function (data, data_anb) {
            return data["region_id"] !== undefined && data["region_id"] === 0
        }
        moveBody(options, myFilter);
}


/**
 * EN: Move something to a separate folder by condition in the code text from the OCR dataset
 * UA: Перенести в окрему папку щось за умовою в тексті коду з OCR-датасету
 * RU: Перенести в отдельную папку что-либо по условию в тексте кода из OCR-датасета
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveSomething \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/train.true/
 */
async function moveSomething (options) {
    const
        myFilter = function (data, data_anb) {
            //return data.region_id != 12 || data.count_lines > 1
            //return data.region_id != 1 || data.count_lines > 1
            //return data.region_id != 1 && data.region_id != 2
            //return data.description.length > 7
            //return data.description.length == 7
            //return data.size.height >= 32
            //return data.description.indexOf("L") != -1
            //return data.description.slice(6) == 'XA'
            return Number(data.count_lines) != 1
            //return Number(data.count_lines) == 2
            //return data.region_id != 2
            //return Number(data.region_id) != 7
            //return Number(data.state_id) != 2
            //return Number(data.region_id) != 3
            //return data.count_lines == undefined || Number(data.count_lines) != 1
        }
        moveBody(options, myFilter);
}


/**
 * EN: Remove all duplicate numbers from a dataset based on another dataset, for example, remove all duplicate numbers
 *     from train that occur in test
 * UA: Видалити всі дублі по номерам з датасету на основі іншого датасету, наприклад видалити всі дублі по номерам
 *     з train які зустрічаються у test
 * RU: Удалить все дубли по номерам из датасета на основе другого датасета, например удалить все дубли по номерам
 *     из train которые встречаются в test
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=removeAllNpDupesThatOccurInAnotherDataset \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example/train/ \
 *                                               --opt.anotherDir=../data/dataset/TextDetector/ocr_example2/test/ \
 *                                               --opt.dupesDir=../data/dataset/TextDetector/ocr_example2/dupes/
 */
async function removeAllNpDupesThatOccurInAnotherDataset (options) {
    const srcDir = options.srcDir || './train',
        anotherDir = options.anotherDir || './test',
        dupesDir = options.dupesDir || './dupes',

        annExt = '.' + config.dataset.ann.ext ,
        anotherAnnPath = path.join(anotherDir, config.dataset.ann.dir),
        anotherAnnFullPath = path.isAbsolute(anotherAnnPath)?anotherAnnPath:path.join(process.cwd(), anotherAnnPath),
        srcFullDir = path.isAbsolute(srcDir)?srcDir:path.join(process.cwd(), srcDir),
        dupesFullDir = path.isAbsolute(dupesDir)?dupesDir:path.join(process.cwd(), dupesDir),
        srcAnnPath = path.join(srcDir, config.dataset.ann.dir),
        srcAnnFullPath = path.isAbsolute(srcAnnPath)?srcAnnPath:path.join(process.cwd(), srcAnnPath),
        anotherNpArr = {}
        //src = { annPath: path.join(srcDir, config.dataset.ann.dir) }

    ;
    let checkedAnn = [],
        checkedImg = []
    ;
    checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
    checkDirStructure(anotherDir, [config.dataset.img.dir,config.dataset.ann.dir], true);
    checkDirStructure(dupesDir, [config.dataset.img.dir,config.dataset.ann.dir], true);
    const anotherAnnFiles = fs.readdirSync(anotherAnnFullPath),
          srcAnnFiles = fs.readdirSync(srcAnnFullPath)
    ;

    for (let anotherAnnFile of anotherAnnFiles) {
        let data = require(path.join(anotherAnnFullPath, anotherAnnFile));
        if (data.description != undefined) {
            anotherNpArr[data.description] = true
        }
    }
    //console.log(anotherNpArr);
    for (let srcAnnFile of srcAnnFiles) {
        let data = require(path.join(srcAnnFullPath, srcAnnFile));
        if (data.description != undefined && anotherNpArr[data.description] != undefined) {
            let imgName = `${data.name}.${config.dataset.img.ext}`
            checkedAnn.push(srcAnnFile);
            checkedImg.push(imgName);
        }
    }
    console.log(`Garbage: ${checkedAnn.length}`);
    // console.log(`srcFullDir: ${srcFullDir}, dupesFullDir: ${dupesFullDir}`);
    // console.log(`checkedAnn.length: ${checkedAnn.length}, checkedImg.length: ${checkedImg.length}`);
    // console.log(`annDir:${config.dataset.ann.dir}, imgDir:${config.dataset.img.dir}`);
    moveDatasetFiles({srcDir:srcFullDir, targetDir:dupesFullDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
}


/**
 * EN: Analyze the OCR dataset and build statistics on the uniqueness of number combinations and remove duplicates
 * UA: Проаналізувати OCR датасет та побудувати статистику по унікальності номерних комбінацій і прибрати дублі
 * RU: Проанализировать OCR датасет и построить статистику по уникальности номерных комбинаций и убрать дубли
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=removeNpDupesFromDataset \
 *                                               --opt.datasetDir=../data/dataset/TextDetector/ocr_example2/ \
 *                                               --opt.reportJson=./dataset_stats.json \
 *                                               --opt.moveDupesDir=../data/dataset/TextDetector/dupes/
 *
 * Тільки перенести дублі
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=removeNpDupesFromDataset \
 *                                               --opt.datasetDir=../data/dataset/TextDetector/ocr_example2/ \
 *                                               --opt.moveDupesDir=../data/dataset/TextDetector/dupes/
 */
async function removeNpDupesFromDataset (options) {
    const datasetDir = options.datasetDir || './data/eu',
        reportJson = options.reportJson, // If "options.reportJson" is not specified, the report will not be generated.
        moveDupesDir = options.moveDupesDir, // If "options.moveDupesDir" is not specified, the duplicate numberplate will not be moved.
        stats = {},
        anomalyStats = {},
        partDirs = {}
    ;

    for (let partDir of config.dataset.partDirs) {
        const
            annExt = '.'+config.dataset.ann.ext,
            annPath = path.join(datasetDir, partDir, config.dataset.ann.dir),
            imgPath = path.join(datasetDir, partDir, config.dataset.img.dir),
            annFiles = fs.readdirSync(annPath),
            annFullPath = path.isAbsolute(annPath)?annPath:path.join(process.cwd(), annPath),
            imgFullPath = path.isAbsolute(imgPath)?imgPath:path.join(process.cwd(), imgPath),
            checkedAnn = [],
            checkedImg = []
        ;
        for (let annFile of annFiles) {
            let data = require(path.join(annFullPath,annFile)),
                imgName = `${data.name}.${config.dataset.img.ext}`;
            if (data.description != undefined) {
                if (stats[data.description] == undefined) {
                    stats[data.description] = {
                        filesFound: new Set()
                    }
                }
            }
            let fileSize = sizeOf(path.join(imgFullPath, imgName));
            delete fileSize.type;
            stats[data.description].filesFound.add({
                part: partDir,
                file: annFile,
                size: fileSize
            })
        }
    }

    // Searching for potential anomaly
    let annExt = `.${config.dataset.ann.ext}`;
    for (let np in stats) {
        if (stats[np].filesFound.size > 1) { // && getPartsCount(stats[np].filesFound) > 1
            let fileItems = Array.from(stats[np].filesFound);
            fileItems = fileItems.sort(function(a, b) { return b.size.width - a.size.width });
            fileItems.shift();
            for (let fileItem of fileItems) {
                let baseName = path.basename(fileItem.file, annExt),
                    imgName = `${baseName}.${config.dataset.img.ext}`
                ;
                if (partDirs[fileItem.part] == undefined ) {
                    partDirs[fileItem.part] = {
                        ann: [],
                        img: []
                    }
                }
                partDirs[fileItem.part].ann.push(fileItem.file);
                partDirs[fileItem.part].img.push(imgName);
            }
            anomalyStats[np] = stats[np]
        }
    }

    console.log(`In the dataset "${datasetDir}", we found ${Object.keys(anomalyStats).length} number plates with duplicates.`)

    if (reportJson != undefined) {
        // Sort anomaly stats
        console.log(`Generating a duplicate report to the ${reportJson} file.`)
        //let reportAnomalyStats = statsTools.sortAnomalyStats(anomalyStats);

        fs.writeFileSync(reportJson, JSON.stringify(anomalyStats, null, 2), 'utf-8');
        console.log(`Done`)
    }

    if (moveDupesDir != undefined) {
        let cnt = 0;
        console.log(`Transferring duplicates from the dataset to the "${moveDupesDir}" directory.`)
        for (let partDir in partDirs) {
            cnt += partDirs[partDir].ann.length;
            console.log(`Move ${partDirs[partDir].ann.length} dupes for "${partDir}" section`)
            let srcDir = path.join(datasetDir, partDir),
                targetDir = path.join(moveDupesDir, partDir)
            ;
            checkDirStructure(targetDir, [config.dataset.img.dir, config.dataset.ann.dir], true);
            moveDatasetFiles({srcDir, targetDir, Anns: partDirs[partDir].ann,
                Imgs: partDirs[partDir].img, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
        }
        console.log(`Done`)
        console.log(`${cnt} duplicates were moved to the "${moveDupesDir}" directory.`)
    }
}



/**
 * EN: Move duplicate records (by photo) to a separate folder
 * UA: Перенести дублікати записай (по фото) в окрему папку
 * RU: Перенести дубликаты записай (по фото) в отдельную папку
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveDupes \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/train.dupes/
 */
async function moveDupes (options) {
    const srcDir = options.srcDir || './draft',
        targetDir = options.targetDir || './checked',
        annExt = '.'+config.dataset.ann.ext,
        src = { annPath: path.join(srcDir, config.dataset.ann.dir) },
        imgPath = path.join(srcDir, config.dataset.img.dir)
    ;
    let checkedAnn = [],
        checkedImg = []
    ;
    let checkSum = {}, checklogs = {};
    checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
    checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);

    fs.readdir(src.annPath, async function(err, items) {
        for (let filename of items) {
            const  fileObj = path.parse(filename);
            if (fileObj.ext === annExt) {
                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                    annFileName = path.join( src.annPath, annName);
                const data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(),
                                                                                        annFileName)),
                    imgName = `${data.name}.${config.dataset.img.ext}`
                ;
                let imgFullFile = path.join(imgPath, imgName),
                    imgSize = sizeOf(imgFullFile),
                    imgMd5 = await md5File(imgFullFile),
                    imgSizeHash = `${imgMd5}-${imgSize.width}x${imgSize.height}`;

                if (checkSum[imgSizeHash] !== undefined) {
                    checkedAnn.push(annName);
                    checkedImg.push(imgName);
                    if (checklogs[checkSum[imgSizeHash]] === undefined) {
                        checklogs[checkSum[imgSizeHash]] = []
                    }
                    checklogs[checkSum[imgSizeHash]].push(imgName)
                } else {
                    checkSum[imgSizeHash] = imgName;
                }
            }
        }
        console.log(`Garbage: ${checkedAnn.length}`);
        moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
        fs.writeFileSync('./logs.json', JSON.stringify(checklogs,null,2));
    });
}


/**
 * EN: Glue several folders into one only for unpainted numbers
 * UA: Склеїти кілька папок в одну лише для незафарбованих номерів
 * RU: Склеить несколько папок в одну только для незакрашеных номеров
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=dataJoin \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/val/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/train/
 */
async function dataJoin (options) {
    if (!Array.isArray(options.srcDir) || options.srcDir.length < 2) {
        throw new Error('"opt.srcDir" must be array for 2 (min) elements!');
    }
    console.log("", options.targetDir);
    const srcDir = options.srcDir,
          targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'),
          annExt = '.'+config.dataset.ann.ext
    ;
    for (let dir of srcDir) {
        checkDirStructure(dir,[config.dataset.img.dir,config.dataset.ann.dir]);
    }
    checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);
    joinAnnotationOcrDataset(srcDir, annExt, targetDir)
}


module.exports = {
    index,
    createAnnotations,
    moveChecked,
    dataSplit,
    mergeDatasets,
    moveGarbage,
    dataJoin,
    moveSomething,
    moveDupes,
    removeNpDupesFromDataset,
    removeAllNpDupesThatOccurInAnotherDataset
};
